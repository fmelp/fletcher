from bs4 import BeautifulSoup as bs
from selenium import webdriver
import csv

urls = ['http://www.tripadvisor.com/Hotel_Review-g35805-d87590-Reviews-Hilton_Chicago-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d609738-Reviews-The_James_Chicago-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d111492-Reviews-Hotel_Monaco_Chicago_a_Kimpton_Hotel-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d236299-Reviews-Sofitel_Chicago_Water_Tower-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d114595-Reviews-MileNorth_A_Chicago_Hotel-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d287568-Reviews-Hard_Rock_Hotel_Chicago-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d87654-Reviews-The_Talbott_Hotel-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d111500-Reviews-Omni_Chicago_Hotel-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d90959-Reviews-Homewood_Suites_by_Hilton_Chicago_Downtown-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d87648-Reviews-Millennium_Knickerbocker_Hotel_Chicago-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d87638-Reviews-Sheraton_Chicago_Hotel_and_Towers-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d114581-Reviews-Swissotel_Chicago-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d90985-Reviews-Hotel_Allegro_Chicago_a_Kimpton_Hotel-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d293203-Reviews-Kinzie_Hotel-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d87620-Reviews-InterContinental_Chicago-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d111501-Reviews-The_Palmer_House_Hilton-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d217498-Reviews-Conrad_Chicago-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d114577-Reviews-Fairmont_Chicago_Millennium_Park-Chicago_Illinois.html',
        'http://www.tripadvisor.com/Hotel_Review-g35805-d87617-Reviews-Hyatt_Regency_Chicago-Chicago_Illinois.html']

def set_up_driver():
    '''
    @return : a PhantomJS webdriver
    '''
    uastring = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1944.0 Safari/537.36'
    dcap = webdriver.DesiredCapabilities.PHANTOMJS
    dcap["phantomjs.page.settings.userAgent"] = uastring
    exec_path = '/usr/local/bin/phantomjs'
    driver = webdriver.PhantomJS(exec_path)
    driver.set_window_size(1024, 768)
    return driver

def get_more_links(urls):
    all_urls = []
    for url in urls:
        loc = url.find("Reviews-")
        h_urls = []
        for i in xrange(1, 61):
            next_page = url[:loc] + 'or' + str(i) + '0-' + url[loc:]
            h_urls.append(next_page)
        all_urls.append(h_urls)
    return all_urls

# driver.find_element_by_css_selector('span.partnerRvw span').click()
def get_reviews(all_urls, driver):
    reviews = []
    for hotel in all_urls:
        print "--------hotel done--------"
        for url in hotel:
            try:
                driver.get(url)
                driver.find_element_by_css_selector('span.partnerRvw span').click()
                soup = bs(driver.page_source)
                divs = soup.find_all(class_='entry')
                for div in divs:
                    if len(div.get_text()) > 400:
                        reviews.append(div.get_text().strip())
            except:
                print 'missed one'
                continue
    return reviews

driver = set_up_driver()
all_urls = get_more_links(urls)
reviews = get_reviews(all_urls, driver)


a = []
for review in reviews:
    a.append([review])


print "writing to csv"
with open('ta_new3.csv', 'w') as f:
    writer = csv.writer(f)
    for row in a:
        try:
            writer.writerow(row)
        except:
            continue