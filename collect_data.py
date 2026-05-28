import pandas as pd
import random
import time
from playwright.sync_api import sync_playwright

def scrape_keyword(page, keyword, max_pages=25):
    print(f"\n=== 開始搜尋關鍵字：{keyword} ===")
    print("前往 104 人力銀行")

    page.goto("https://www.104.com.tw/jobs/main/")
    time.sleep(random.uniform(2, 4))

    search_input = page.locator('input[data-gtm-index="搜尋欄位-搜尋點擊"]')
    search_input.clear()
    search_input.fill(keyword)
    time.sleep(random.uniform(0.5, 1.5))
    page.locator('button[data-gtm-index="搜尋欄位-搜尋點擊-送出-工作快訊"]').click()
    print("等待搜尋結果頁面載入......")

    page.wait_for_selector("div.col.main", timeout=20000)
    time.sleep(3)

    keyword_data = []

    for current_pages in range(1, max_pages + 1):
        print(f"\n正在搜尋包含關鍵字 [{keyword}] 的相關職缺")

        for s in range(4):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(random.uniform(2.5, 4.0))

        job_cards = page.locator("div[class*='job-list-container']").all()
        print(f"偵測到當前頁面有 {len(job_cards)} 個職缺卡片，開始提取")

        for card in job_cards:
            try:
                title = card.locator("a[data-gtm-joblist='職缺-職缺名稱']").get_attribute("title")
                salary = card.locator("a[data-gtm-joblist*='薪資']").inner_text()
                edu_LV = card.locator("a[data-gtm-joblist*='職缺-學歷']").inner_text()

                keyword_data.append({
                    "職稱": title, 
                    "薪資原始字串": salary, 
                    "職缺所需學歷": edu_LV
                })
                
            except Exception:
                continue

        if current_pages < max_pages:
            try:
                next_btn = page.locator("button.js-next-page, a.js-next-page, button:has-text('下一頁'), a:has-text('下一頁')").first

                if next_btn.is_visible() and next_btn.is_enabled():
                    print("點擊下一頁")
                    next_btn.click()
                    page.wait_for_load_state("networkidle", timeout=10000)
                    time.sleep(random.uniform(2.0, 3.5))
                else:
                    print("找不到下一頁按鈕或已達末頁，停止抓取此關鍵字的職缺")
                    break
            except Exception as e:
                print(f"翻頁嘗試失敗: {e}，結束採集此關鍵字的職缺")
                break
        
    return keyword_data


def run_scraper():
    keywords = ["AI工程師", "機器學習", "資料科學家", "深度學習"]
    all_collected_data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1000}
        )
        page = context.new_page()
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        for kw in keywords:
            kw_data = scrape_keyword(page, kw, max_pages=15)
            all_collected_data.extend(kw_data)
            print(f"關鍵字 [{kw}] 採集完畢，目前累計總原始數據：{len(all_collected_data)} 筆")
            time.sleep(random.uniform(5, 10))

        if all_collected_data:
            df = pd.DataFrame(all_collected_data)
            df.to_csv("ai_jobs_raw.csv", index=False, encoding="utf-8-sig")
            print(f"\n共 {len(df)} 筆原始數據，已寫入 ai_jobs_raw.csv")
        else:
            print("\n發生錯誤，未能抓取任何數據")

        browser.close()

if __name__ == "__main__":
    run_scraper()