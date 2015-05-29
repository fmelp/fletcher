from bs4 import BeautifulSoup as bs
import re
from urllib2 import urlopen as getpage
import json



urls = ["http://www.expedia.com/Chicago-Hotels-Hilton-Chicago-Michigan-Ave-Cultural-Mile.h12570.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-The-James-Chicago.h26728.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Hotel-Monaco-Chicago.h4200.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Sofitel-Chicago-Water-Tower.h864219.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-MileNorth-Hotel.h21619.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Hard-Rock-Hotel-Chicago.h982552.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-The-Talbott-Hotel.h23963.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Omni-Chicago-Hotel.h11076.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Homewood-Suites-By-Hilton-Chicago-Downtown.h281523.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Millennium-Knickerbocker.h15087.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Sheraton-Chicago-Hotel-And-Towers.h12167.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Swissotel-Chicago.h17288.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Hotel-Allegro.h34496.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Kinzie-Hotel.h992500.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-InterContinental-Chicago-Magnificent-Mile.h20205.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-The-Palmer-House-Hilton.h8079.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Conrad-Chicago.h558027.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Fairmont-Chicago-At-Millennium-Park.h15937.Hotel-Reviews",
        "http://www.expedia.com/Chicago-Hotels-Hyatt-Regency-Chicago.h4903.Hotel-Reviews"]

hotel_names = ["hilton", "james", "monaco", "sofitel", "milenorth",
                "hardrock", "talbott", "omni", "homewood", "millennium",
                "sheraton", "swissotel", "allegro", "kinzie",
                "intercontiental", "palmer", "conrad", "fairmont", "hyatt"]

def get_more_pages(url):
    pages_urls = [url]
    for i in xrange(2,30):
        index = url.rfind('.')
        new = url[:index] + '-p' + str(i) + url[index:]
        pages_urls.append(new)
    return pages_urls

def get_review_text(urls):
    review_d = {}
    for i, url in enumerate(urls):
        pages = get_more_pages(url)
        reviews = []
        for page in pages:
            try:
                source = getpage(page)
            except:
                print hotel_names[i], ' missed'
                continue
            soup = bs(source)
            divs = soup.find_all(class_="review-text")
            for div in divs:
                reviews.append(div.contents[0])
        review_d[hotel_names[i]] = reviews
        print hotel_names[i], ' DONE'
    return review_d

def save_to_json_file(dic):
    fname = 'expedia_data2.json'
    with open(fname, 'w') as fout:
        json.dump(dic, fout, indent=2)


save_to_json_file(get_review_text(urls))
