import pandas as pd
from faker import Faker
import random

fake = Faker()

def generate_data(num_records=50):
    data = []
    for _ in range(num_records):
        account_number = fake.bban()
        customer_name = fake.name()
        phone_number = fake.phone_number()
        
        # Generate a few transactions for each customer
        for _ in range(random.randint(1, 5)):
            date = fake.date_this_year()
            transaction_id = fake.uuid4()
            description = random.choice([
                "Grocery Store", "Online Retailer", "Utility Bill", 
                "Salary Deposit", "Restaurant", "Gas Station", "Transfer Out"
            ])
            amount = round(random.uniform(-500, 2000), 2)
            balance = round(random.uniform(1000, 50000), 2)
            
            data.append({
                "Date": date,
                "Transaction ID": transaction_id,
                "Description": description,
                "Amount": amount,
                "Balance": balance,
                "Account Number": account_number,
                "Customer Name": customer_name,
                "Phone Number": phone_number
            })
            
    df = pd.DataFrame(data)
    df.to_csv("bank_statement.csv", index=False)
    print(f"Generated {len(df)} records in bank_statement.csv")

if __name__ == "__main__":
    generate_data()
