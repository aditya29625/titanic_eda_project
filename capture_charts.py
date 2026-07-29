from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_viewport_size({"width": 1280, "height": 800})
    page.goto('http://localhost:8502')
    
    # Wait for the Streamlit app to load completely
    time.sleep(5)
    
    # Scroll down slightly to center the top two charts
    page.evaluate('window.scrollTo(0, 450)')
    time.sleep(1)
    page.screenshot(path='charts_1_2.png')
    
    # Scroll down further to center the bottom two charts
    page.evaluate('window.scrollTo(0, 1100)')
    time.sleep(1)
    page.screenshot(path='charts_3_4.png')
    
    browser.close()
