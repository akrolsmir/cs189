import xml.etree.ElementTree as ET
import urllib.request
import time

# anime_by_rankings.xml is available at:
# http://www.animenewsnetwork.com/encyclopedia/reports.xml?id=172&nlist=all
tree = ET.parse('anime_by_ratings.xml')
root = tree.getroot()

# Download details for blocks of 40 anime at a time
for i in range(2440, 4680, 40):
    print(i)
    ids = [item.attrib['id'] for item in root[i:i+40]]
    query = 'http://cdn.animenewsnetwork.com/encyclopedia/api.xml?title=' + '/'.join(ids)
    urllib.request.urlretrieve(query, 'anime/{0}.xml'.format(i))
    time.sleep(60)