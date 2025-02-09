class Order:
    def __init__(self):
        self.items = []
        self.quantities = []
        self.prices = []
        self.status = "open"

    def add_item(self,name:str,quantity:int,price:float) -> None:
        self.items.append(name)
        self.quantities.append(quantity)
        self.prices.append(price)

class PaymentProcessor:
    def pay(self,order:Order, security_code: str, payment_type: str):
        if payment_type == "debit":
            print("Processing debit payment type")
            print(f"Verifying security code: {security_code}")
            order.status = "paid"
        elif payment_type == "credit":
            print("Processing credit payment type")
            print(f"Verifying security code: {security_code}")
            order.status = "paid"
        else:
            raise Exception(f"Unknown payment type: {payment_type}")

class CalculateProcessor:
    def total_price(self,order:Order):
        total = 0
        for quantity,price in zip(order.quantities, order.prices):
            total += quantity * price
        return total

order = Order()
order.add_item("Laptop", 1, 150)
order.add_item("SSD", 1, 50)
order.add_item("USB Cable", 1, 15)

processor = PaymentProcessor()
processor.pay(order, "12345", "debit")
print(order.status)

total = CalculateProcessor()
print(total.total_price(order))