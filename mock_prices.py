import csv
import random

input_file = 'data/flipkart_product_review.csv'
output_file = 'data/flipkart_product_review.csv'

# Read existing data
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# Add 'price' to fieldnames if not exists
if 'price' not in fieldnames:
    fieldnames.insert(2, 'price')

# Generate dummy prices based on product title
product_prices = {}

for row in rows:
    title = row['product_title']
    if title not in product_prices:
        # Give some realistic dummy prices based on brand
        if 'OnePlus' in title:
            product_prices[title] = random.choice([1999, 2199])
        elif 'realme Buds Wireless' in title:
            product_prices[title] = random.choice([1499, 1799])
        elif 'realme Buds Q' in title:
            product_prices[title] = random.choice([1599, 1999])
        elif 'realme Buds 2' in title:
            product_prices[title] = random.choice([599, 699])
        elif 'BoAt Airdopes' in title:
            product_prices[title] = random.choice([1299, 1499])
        elif 'BoAt Rockerz' in title:
            product_prices[title] = random.choice([999, 1299])
        elif 'U&I' in title:
            product_prices[title] = random.choice([499, 799])
        else:
            product_prices[title] = random.randint(500, 3000)
            
    row['price'] = product_prices[title]

# Write back
with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Added dummy prices to the dataset successfully!")
