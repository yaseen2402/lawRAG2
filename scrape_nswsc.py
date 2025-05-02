import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import re
import os

# --- Configuration ---
BASE_URL = "https://www.caselaw.nsw.gov.au"
# Using the browse URL structure you identified
START_URL_TEMPLATE = "https://www.caselaw.nsw.gov.au/browse?display=all#sort:decisionDate,desc,decisionDate,desc;courts:54a634063004de94513d8281;years:2025,2024,2023,2022,2021,2020;startsWith:a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,0-9;page:{page_num}"
MAX_PAGES_TO_SCRAPE = 41  # Scrape pages 0 to 40
OUTPUT_FILE = "nsw_supreme_court_cases_selenium.jsonl"
LOAD_WAIT_TIME = 20  # Max seconds to wait for elements/page transitions
NAVIGATION_DELAY = 5 # Seconds to wait after clicking 'Next'
REQUEST_DELAY = 5    # Seconds between individual case detail fetch requests (using requests)
USER_AGENT = 'LegalResearch/1.0 (Purpose: Academic legal research - respecting terms)'

# --- Helper Functions ---

def setup_driver():
    options = webdriver.ChromeOptions()
#     options.add_argument("--headless") # Uncomment to run without a visible browser window
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080") # Can sometimes help with element visibility
    options.add_argument(f"user-agent={USER_AGENT}")

    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
        print("WebDriver setup complete.")
        return driver
    except Exception as e:
        print(f"Error setting up WebDriver: {e}")
        print("Please ensure Google Chrome is installed and webdriver-manager can download chromedriver.")
        return None

def get_soup_requests(url):
    """Fetches a URL using requests and returns a BeautifulSoup object."""
    headers = {'User-Agent': USER_AGENT}
    try:
        # Add a small delay before *every* requests call
        time.sleep(REQUEST_DELAY)
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except requests.exceptions.RequestException as e:
        print(f"    Error fetching {url} with requests: {e}")
        return None

def extract_case_details_requests(case_url):
    """Extracts details from a single case page using requests."""
    soup = get_soup_requests(case_url)
    if not soup:
        return None

    details = {"url": case_url}
    try:
        # Find the judgment div that contains all content
        judgment_div = soup.find('div', class_='judgment')
        if not judgment_div:
            return None

        # Extract metadata from coversheet section
        coversheet = judgment_div.find('div', class_='coversheet')
        if coversheet:
            dl_items = coversheet.find('dl', class_='dl-horizontal')
            if dl_items:
                # Process all dt/dd pairs
                for dt in dl_items.find_all('dt'):
                    field_name = dt.get_text(strip=True).rstrip(':').lower().replace(' ', '_')
                    dd = dt.find_next('dd')
                    if dd:
                        # Handle special formatting for certain fields
                        if field_name in ['catchwords', 'legislation_cited', 'cases_cited', 'decision']:
                            # Get all paragraph texts for these fields
                            field_value = [p.get_text(strip=True) for p in dd.find_all('p')]
                        else:
                            field_value = dd.get_text(strip=True)
                        details[field_name] = field_value

        # Extract judgment content
        body = judgment_div.find('div', class_='body')
        if body:
            paragraphs = []
            
            # Handle both h1 and h2 headings
            headings = []
            for h in body.find_all(['h1', 'h2']):
                heading_text = h.get_text(strip=True)
                if heading_text:
                    headings.append({
                        "level": int(h.name[1]),
                        "text": heading_text
                    })
            details['headings'] = headings

            # Extract numbered paragraphs
            for ol in body.find_all('ol', class_='num1'):
                # Get the starting number from start attribute or style
                start_num = 1
                if 'start' in ol.attrs:
                    start_num = int(ol['start'])
                elif 'style' in ol.attrs:
                    counter_match = re.search(r'counter-reset:li-counter (\d+)', ol['style'])
                    if counter_match:
                        start_num = int(counter_match.group(1)) + 1

                for idx, li in enumerate(ol.find_all('li', recursive=False)):
                    p_num = start_num + idx
                    text = li.get_text(strip=True)
                    
                    # Check if paragraph has footnotes/endnotes
                    footnotes = []
                    for sup in li.find_all('sup'):
                        if sup.find('a'):
                            note_text = sup.get_text(strip=True)
                            footnotes.append(note_text)
                    
                    para_data = {
                        "p_num": p_num,
                        "text": text
                    }
                    if footnotes:
                        para_data["footnotes"] = footnotes
                    
                    paragraphs.append(para_data)

            details['paragraphs'] = paragraphs
            details['full_text'] = body.get_text(strip=True)

            # Extract endnotes if present
            endnotes_div = judgment_div.find('div', class_='endnote-container')
            if endnotes_div:
                endnotes = []
                for p in endnotes_div.find_all('p'):
                    note_text = p.get_text(strip=True)
                    if note_text:
                        endnotes.append(note_text)
                details['endnotes'] = endnotes

        # Basic validation
        required_fields = ['medium_neutral_citation', 'paragraphs']
        missing_fields = [f for f in required_fields if not details.get(f)]
        if missing_fields:
            print(f"Warning: Missing critical fields: {', '.join(missing_fields)} for {case_url}")

        return details

    except Exception as e:
        print(f"Error parsing {case_url}: {e}")
        return None


def parse_case_list_selenium(driver):
    """Gets page source from selenium driver and parses case links."""
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    case_links = []
    case_details = []

    # Find the main container that holds all results
    results_container = soup.find('div', id='decisionList')
    if not results_container:
        print("    Warning: Could not find results container (decisionList).")
        return []

    # Process each case result
    for result_div in results_container.find_all('div', class_='row result'):
        try:
            # Get content div that contains case name and link
            content_div = result_div.find('div', class_='col-sm-8 cntn')
            if not content_div:
                continue

            # Get case link and title
            link_tag = content_div.find('a')
            if not link_tag or not link_tag.get('href'):
                continue
                
            case_url = BASE_URL + link_tag['href']
            case_name = link_tag.get_text(strip=True)

            # Get catchwords if available
            catchwords = None
            catchwords_p = content_div.find('p', string='Catchwords:')
            if catchwords_p and catchwords_p.find_next('p'):
                catchwords = catchwords_p.find_next('p').get_text(strip=True)

            # Get metadata from info column
            info_div = result_div.find('div', class_='info')
            judge = None
            decision_date = None
            
            if info_div:
                # Extract judge name
                judge_item = info_div.find('li', string='Judgment of')
                if judge_item and judge_item.find_next('li'):
                    judge = judge_item.find_next('li').get_text(strip=True)
                
                # Extract decision date  
                date_item = info_div.find('li', string='Decision date')
                if date_item and date_item.find_next('li'):
                    decision_date = date_item.find_next('li').get_text(strip=True)

            # Store case metadata
            case_data = {
                'url': case_url,
                'case_name': case_name,
                'catchwords': catchwords,
                'judge': judge,
                'decision_date': decision_date
            }
            
            case_details.append(case_data)
            case_links.append(case_url)

        except Exception as e:
            print(f"    Error parsing case result: {e}")
            continue

    if not case_links:
        print("    Warning: No case links found on page.")
    else:
        print(f"    Found {len(case_links)} cases on page.")

    return case_links

def click_page_number(driver, wait, page_num):
    try:
        print(f"Finding and clicking page {page_num} link...")
        
        selector = f'a.page-link[href="{page_num}"]'
        page_link = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
        page_link.click()

        # Wait until span.current shows the correct page number
        wait.until(EC.presence_of_element_located((By.XPATH, f'//span[@class="current" and text()="{page_num}"]')))
        
        print(f"Successfully navigated to page {page_num}!")
        time.sleep(5)  # optional small wait
        return True

    except Exception as e:
        print(f"Error clicking page {page_num}: {e}")
        return False

def navigate_to_next_page_selenium(driver, wait, next_page_zero_indexed):
    """Navigates directly by modifying the page number in the URL."""
    try:
        next_url = START_URL_TEMPLATE.format(page_num=next_page_zero_indexed)
        print(f"    Navigating directly to: {next_url}")
        driver.get(next_url)
        
        # Wait for page load
        if not wait_for_page_load(driver, wait):
            print("    Warning: Page did not load properly after navigation.")
            return False
        
        return True
    except Exception as e:
        print(f"    Error navigating directly to page {next_page_zero_indexed}: {e}")
        return False


def wait_for_page_load(driver, wait):
    """Wait for critical elements to load."""
    try:
        # Wait for document ready state
        wait.until(lambda d: d.execute_script('return document.readyState') == 'complete')
        
        # Wait for results container with increased timeout
        wait.until(EC.presence_of_element_located((By.ID, "decisionList")))
        
        # Extra wait for dynamic content
        time.sleep(5)
        return True
    except TimeoutException:
        print("Page failed to load completely")
        return False
    
OUTPUT_FILE_TEMPLATE = "cases_{page_num}.jsonl"

# --- Main Scraping Logic ---
def main():
    # Initialize driver
    driver = setup_driver()
    if not driver:
        return

    # Increase wait time for slow pages
    wait = WebDriverWait(driver, 30)  # Increased from 20
    processed_case_urls = set()

    # Load previously processed URLs
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        processed_case_urls.add(data.get('url', ''))
                    except (json.JSONDecodeError, KeyError):
                        continue
            print(f"Loaded {len(processed_case_urls)} previously processed URLs.")
        except Exception as e:
            print(f"Warning: Could not load processed URLs: {e}")

    try:
        print("Loading initial page...")
        driver.get(START_URL_TEMPLATE.format(page_num=2))
        
        if not wait_for_page_load(driver, wait):
            print("Failed to load initial page. Exiting.")
            return

        # Rest of main() remains unchanged
        current_page = 2
        click_page_number(driver, wait, current_page)
        while current_page < MAX_PAGES_TO_SCRAPE:
            print(f"\n--- Processing List Page: {current_page} ---")

            # Wait for the main results area to appear
            try:
                # --- !!! USER ACTION REQUIRED: Inspect & Verify Selector !!! ---
                results_area_selector = "div.row.result" # Adjust selector
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, results_area_selector)))
                print(f"List page {current_page} loaded.")
                # Add a small extra wait just in case content loads dynamically after container appears
                time.sleep(5)
            except TimeoutException:
                print(f"Timed out waiting for results area on page {current_page}. Trying to navigate.")
                # Attempt to navigate away from potentially stuck page
                if not navigate_to_next_page_selenium(driver, wait, current_page):
                     print("Failed to navigate away from stuck page. Stopping.")
                     break
                current_page += 1
                continue # Skip processing this page

            # Parse case URLs from the current list page
            case_urls_on_page = parse_case_list_selenium(driver)
            print(f"Found {len(case_urls_on_page)} case URLs on page {current_page}.")

            if not case_urls_on_page and current_page > 0:
                 print("Found no case URLs on this page, assuming end of results.")
                 # break # Option: stop if a page yields no results

            new_urls_found = 0
            # Extract details for each case URL found
            for case_url in case_urls_on_page:
                if case_url in processed_case_urls:
                    # print(f"  Skipping already processed URL: {case_url}")
                    continue

                new_urls_found += 1
                print(f"  Fetching details for new URL: {case_url}")
                case_data = extract_case_details_requests(case_url) # Using requests helper

                if case_data:
                    # Save data immediately
                    try:
                        output_file = OUTPUT_FILE_TEMPLATE.format(page_num=current_page)
                        with open(output_file, 'a', encoding='utf-8') as f:
                            json.dump(case_data, f, ensure_ascii=False)
                            f.write('\n')
                        print(f"    Successfully processed and saved: {case_data.get('citation', case_url)}")
                        processed_case_urls.add(case_url) # Mark as processed only on success
                    except IOError as e:
                        print(f"    Error writing to file: {e}")
                    except Exception as e:
                         print(f"    Error processing/saving data for {case_url}: {e}")
                else:
                    print(f"    Failed to extract details for {case_url}")

            print(f"Processed {new_urls_found} new URLs on this page.")

            # Navigate to the next page
            current_page += 1
            if current_page >= MAX_PAGES_TO_SCRAPE:
                print("\nReached maximum configured pages limit.")
                break
            
            success = click_page_number(driver, wait, current_page)

            if not success:
                print(f"Failed to click page {current_page}. Exiting.")
                break

            print(f"Successfully navigated to list page {current_page} (or initiated load).")

    except Exception as e:
        print(f"\n--- An unexpected error occurred during scraping ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
    finally:
        if driver:
            driver.quit()
            print("\nBrowser closed.")
            print(f"Total unique cases processed and saved in this run (approx): {len(processed_case_urls)}") # This count might be slightly off if script loaded previous data
            print(f"Data saved in: {OUTPUT_FILE_TEMPLATE}")

# --- Execution ---
if __name__ == "__main__":
    main()