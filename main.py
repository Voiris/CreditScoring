import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)

df = pd.read_csv("credit.csv")


df = df.drop(["checking_balance", "savings_balance", "default", "property", "installment_plan", "dependents", "other_debtors", "residence_history"], axis=1)

def apply_amount(amount):
    return amount * 92

# ['critical', 'repaid', 'delayed', 'fully repaid',
#        'fully repaid this bank']
def apply_credit_history(credit_history):
    if credit_history == "critical" or credit_history == "delayed":
        return 0
    else:
        return 1

# ['radio/tv', 'education', 'furniture', 'car (new)', 'car (used)',
#        'business', 'domestic appliances', 'repairs', 'others',
#        'retraining']
def apply_purpose(purpose):
    if purpose in ["education", "business"]:
        return 1
    else:
        return 0

# ['> 7 yrs', '1 - 4 yrs', '4 - 7 yrs', 'unemployed', '0 - 1 yrs']
def apply_employment_length(employment_length):
    if employment_length == '> 7 yrs':
        return 4
    elif employment_length == '4 - 7 yrs':
        return 3
    elif employment_length == '1 - 4 yrs':
        return 2
    elif employment_length == '0 - 1 yrs':
        return 1
    elif employment_length == 'unemployed':
        return 0

# ['single male', 'female', 'divorced male', 'married male']
def apply_personal_status(personal_status):
    if personal_status == 'female':
        return 2
    elif personal_status == 'married male' or personal_status == 'divorced male':
        return 1
    elif personal_status == 'single male':
        return 0

# ['own', 'for free', 'rent']
def apply_housing(housing):
    if housing == 'rent':
        return 2
    elif housing == 'for free':
        return 1
    elif housing == 'own':
        return 0

# apply_existing_credits => [1, 2, 3, 4]

# ['yes', 'none']
def apply_telephone(telephone):
    if telephone == 'yes':
        return 1
    else:
        return 0

# ['yes', 'no']
def apply_foreign_worker(foreign_worker):
    if foreign_worker == 'yes':
        return 1
    else:
        return 0

# ['skilled employee', 'unskilled resident',
#        'mangement self-employed', 'unemployed non-resident']
def apply_job(foreign_worker):
    if foreign_worker == 'skilled employee':
        return 3
    elif foreign_worker == 'unskilled resident':
        return 2
    elif foreign_worker == 'mangement self-employed':
        return 1
    elif foreign_worker == 'unemployed non-resident':
        return 0

df["amount"] = df["amount"].apply(apply_amount)
df["credit_history"] = df["credit_history"].apply(apply_credit_history)
df["purpose"] = df["purpose"].apply(apply_purpose)
df["employment_length"] = df["employment_length"].apply(apply_employment_length)
df["personal_status"] = df["personal_status"].apply(apply_personal_status)
df["housing"] = df["housing"].apply(apply_housing)
df["telephone"] = df["telephone"].apply(apply_telephone)
df["foreign_worker"] = df["foreign_worker"].apply(apply_foreign_worker)
df["job"] = df["job"].apply(apply_job)

# Machine Learning
x = df.drop('credit_history', axis=1)
y = df['credit_history']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors=50)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

percent = accuracy_score(y_test, y_pred) * 100

# True Repaid, True Delayed, False Repaid, False Delayed
TR, TD, FR, FD = 0, 0, 0, 0

for test, pred in zip(y_test, y_pred):
    if test - pred == 0:
        if test == 1:
            TR += 1
        else:
            TD += 1
    else:
        if test == 1:
            FD += 1
        else:
            FR += 1

print('Верный прогноз: погашенные -', TR, 'просроченные -', TD)
print('Ошибочный прогноз: погашенные -', FR, 'просроченные -', FD)
print(f"Точность: {percent} %")
