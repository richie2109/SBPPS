import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

placement = pd.read_csv("placementdata.csv")
placement.head()
placement_copy = placement.copy()
print(placement_copy.dtypes)
print(placement_copy.isnull().sum())

placement_copy.drop(['RegNo.'], axis=1, inplace=True)

sns.heatmap(placement_copy.corr())
placement_copy.hist(bins=10,figsize=(10,15))
plt.show()

print(placement_copy.head())

plt.figure(figsize=(20, 15))

ax = plt.subplot(331)
plt.boxplot(placement_copy['Quants'])
ax.set_title('Quants')

ax = plt.subplot(332)
plt.boxplot(placement_copy['LogicalReasoning'])
ax.set_title('Logical Reasoning')

ax = plt.subplot(333)
plt.boxplot(placement_copy['Verbal'])
ax.set_title('Verbal')

ax = plt.subplot(334)
plt.boxplot(placement_copy['Programming'])
ax.set_title('Programming')

ax = plt.subplot(335)
plt.boxplot(placement_copy['Networking'])
ax.set_title('Networking')

ax = plt.subplot(336)
plt.boxplot(placement_copy['CloudComp'])
ax.set_title('CloudComp')

ax = plt.subplot(337)
plt.boxplot(placement_copy['WebServices'])
ax.set_title('WebServices')

ax = plt.subplot(338)
plt.boxplot(placement_copy['DataAnalytics'])
ax.set_title('DataAnalytics')

ax = plt.subplot(339)
plt.boxplot(placement_copy['QualityAssurance'])
ax.set_title('QualityAssurance')

plt.figure(figsize=(20, 15))

ax = plt.subplot(331)
plt.boxplot(placement_copy['AI'])
ax.set_title('AI')

Q1 = placement_copy['Networking'].quantile(0.30)
Q3 = placement_copy['Networking'].quantile(0.70)
IQR = Q3-Q1

filters = (placement_copy['Networking'] >= Q1 - 1.5 * IQR) & (placement_copy['Networking'] <= Q3 + 1.5*IQR)
placement_filtered = placement_copy.loc[filters]
plt.boxplot(placement_filtered['Networking'])

Q1 = placement_copy['CloudComp'].quantile(0.35)
Q3 = placement_copy['CloudComp'].quantile(0.65)
IQR = Q3 - Q1

filters = (placement_copy['CloudComp'] >= Q1 - 1.5 * IQR) & (placement_copy['CloudComp'] <= Q3 + 1.5*IQR)
placement_filtered = placement_copy.loc[filters]
plt.boxplot(placement_filtered['CloudComp'])

Q1 = placement_copy['WebServices'].quantile(0.30)
Q3 = placement_copy['WebServices'].quantile(0.70)
IQR = Q3 - Q1

filters = (placement_copy['WebServices'] >= Q1 - 1.5 * IQR) & (placement_copy['WebServices'] <= Q3 + 1.5*IQR)
placement_filtered = placement_copy.loc[filters]
plt.boxplot(placement_filtered['WebServices'])

Q1 = placement_copy['DataAnalytics'].quantile(0.30)
Q3 = placement_copy['DataAnalytics'].quantile(0.70)
IQR = Q3 - Q1

filters = (placement_copy['DataAnalytics'] >= Q1 - 1.5 * IQR) & (placement_copy['DataAnalytics'] <= Q3 + 1.5*IQR)
placement_filtered = placement_copy.loc[filters]
plt.boxplot(placement_filtered['DataAnalytics'])

Q1 = placement_copy['QualityAssurance'].quantile(0.30)
Q3 = placement_copy['QualityAssurance'].quantile(0.70)
IQR = Q3 - Q1

filters = (placement_copy['QualityAssurance'] >= Q1 - 1.5 * IQR) & (placement_copy['QualityAssurance'] <= Q3 + 1.5*IQR)
placement_filtered = placement_copy.loc[filters]
plt.boxplot(placement_filtered['QualityAssurance'])

X = placement_filtered.drop(['Placed'], axis=1)
y = placement_filtered['Placed']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
logreg = LogisticRegression(max_iter=500, solver='lbfgs')  # Increase max_iter for convergence
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Logistic Regression Metrics
print("Logistic Regression Accuracy:", logreg.score(X_test, y_test))

# Confusion Matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred)
print("Logistic Regression Confusion Matrix:\n", cm)

tn, fp, fn, tp = confusion_matrix(list(y_test), list(y_pred), labels=[0, 1]).ravel()

print('True Positive:', tp)
print('True Negative:', tn)
print('False Positive:', fp)
print('False Negative:', fn)
acc = (tp+tn) / (tp+tn+fn+fp)
print(acc)
precision = tp / (tp+fp)
recall = tp / (tp+fn)
print('Precision: %.3f' % precision)
print('Recall: %.3f' % recall)


# Plot Confusion Matrix for Logistic Regression
fig, ax = plt.subplots(figsize=(4, 4))
ax.matshow(cm, cmap=plt.cm.Purples, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='x-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix - Logistic Regression', fontsize=18)
plt.show()

# Random Forest Classifier
rt = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rt.fit(X_train, y_train)
y_pred1 = rt.predict(X_test)

# Random Forest Metrics
print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred1))

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred1)
print("Random Forest Confusion Matrix:\n", cm_rf)

tn, fp, fn, tp = confusion_matrix(list(y_test), list(y_pred1), labels=[0, 1]).ravel()

print('True Positive:', tp)
print('True Negative:', tn)
print('False Positive:', fp)
print('False Negative:', fn)

acc = (tp+tn) / (tp+tn+fn+fp)
print(acc)
precision = tp / (tp+fp)
recall = tp / (tp+fn)
print('Precision: %.3f' % precision)
print('Recall: %.3f' % recall)

# Plot Confusion Matrix for Random Forest
fig, ax = plt.subplots(figsize=(4, 4))
ax.matshow(cm_rf, cmap=plt.cm.Purples, alpha=0.3)
for i in range(cm_rf.shape[0]):
    for j in range(cm_rf.shape[1]):
        ax.text(x=j, y=i, s=cm_rf[i, j], va='center', ha='center', size='x-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix - Random Forest', fontsize=18)
plt.show()
#decision tree
dt = DecisionTreeClassifier(criterion='gini', max_depth=3)

dt = dt.fit(X_train, y_train)
y_pred2 = dt.predict(X_test)
print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred2)) #print decision tree accuracy

cm_dt = confusion_matrix(y_test, y_pred2)

print("Decision Tree Confusion Matrix:\n", cm_dt)
tn, fp, fn, tp = confusion_matrix(list(y_test), list(y_pred2), labels=[0, 1]).ravel()

print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
acc = (tp+tn) / (tp+tn+fn+fp)
print('Accuracy: %.3f' % acc)
precision = tp / (tp+fp)
recall = tp / (tp+fn)
print('Precision: %.3f' % precision)
print('Recall: %.3f' % recall)

fig, ax = plt.subplots(figsize=(4, 4))
ax.matshow(cm, cmap=plt.cm.Purples, alpha=0.3)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='x-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix - Decision Tree', fontsize=18)
plt.show()

res = logreg.predict([[15.69, 16.09, 16.01, 15.95, 8.07, 6.58, 7.73, 7.79, 6.73, 6.72, 6.79]])
print(res)

res = logreg.predict([[12, 5, 6, 7, 5.07, 6.58, 5, 6, 7, 4, 7]])
print(res)

mean = X.mean(axis='index')
skills = pd.DataFrame({'skills': mean.index, 'list': mean.values})
print(mean)
print(len(skills))

def Predict():
    total = len(skills)
    values = [float(input1value.get()), float(input2value.get()), float(input3value.get()), float(input4value.get()), float(input5value.get()), float(input6value.get()), float(input7value.get()), float(input8value.get()), float(input9value.get()), float(input10value.get()), float(input11value.get())]
    # values = [8, 78,8,6,6.6,6,8,7,8,7]
    # val= [15,16,16,15,8,6,7,7,6,6,6]
    result = logreg.predict([values])
    print(result)
    if result == 0:
        sub = []
        i = 0
        while i < total:
            if values[i] < mean[i]:
                if i != 4:
                    sub.append(skills._get_value(i, 'skills'))
            i += 1
        # t3.insert(END, "Sorry!! you can't be placed..\nYou need to improve your skills on: ")
        print("Sorry!! you can't be placed..\nYou need to improve your skills on:")
        t3.delete("1.0", END)
        myLabel.configure(text="\nSorry!! you can't be placed..\nYou need to improve your skills on:\n ")
        for x in sub:
            print(x)
            t3.insert(END, x + "\n")
    else:
        t3.delete("1.0", END)
        # t3.insert(END, "Congrats!! You can be placed.. Be consistent")
        myLabel.configure(text="Congrats!! You can be placed.. Be consistent")
        print("Congrats!! You can be placed.. Be consistent")


prev_win = None


def message():
    if (input1value.get() == " " and input2value.get() == " " and input3value.get() == " " and input4value.get() == " " and input5value.get() == " " and input6value.get() == " " and input7value.get() == " " and input8value.get() == " " and input9value.get() == " " and input10value.get() == " " and input11value.get() == " "):
        messagebox.showinfo("OPPS!!", "ENTER  VALUES PLEASE")
    else:
        Predict()


def reset():
    global prev_win
    input1value.set(" ")
    input2value.set(" ")
    input3value.set(" ")
    input4value.set(" ")
    input5value.set(" ")
    input6value.set(" ")
    input7value.set(" ")
    input8value.set(" ")
    input9value.set(" ")
    input10value.set(" ")
    input11value.set(" ")
    # myLabel.delete("1.0", END)
    myLabel.config(text="")
    t3.delete("1.0", END)
    t3.insert(' ')
    try:
        prev_win.destroy()
        prev_win = None
    except AttributeError:
        pass


def Exit():
    qexit = messagebox.askyesno("System", "Do you want to exit the system")

    if qexit:
        exit()
        root.destroy()

# Function to validate user input for floating-point numbers
def validate_input(value):
    if value == "":  # Allow clearing the input
        return True
    try:
        num = float(value)  # Attempt to convert the input to a float
        return 1 <= num <= 10  # Return True if the value is between 1 and 10
    except ValueError:
        return False  # Reject the input if it cannot be converted to a float


def validate_input_10_to_20(value):
    if value == "":  # Allow clearing the input
        return True
    try:
        num = float(value)  # Attempt to convert the input to a float
        return 0 < num <= 20  # Return True if the value is within the range
    except ValueError:
        return False  # Reject the input if it cannot be converted to a float


def validate_input_0_to_10(value):
    if value == "":  # Allow clearing the input
        return True
    try:
        num = float(value)  # Attempt to convert the input to a float
        return num <= 10  # Return True if the value is within the range
    except ValueError:
        return False  # Reject the input if it cannot be converted to a float





root = tk.Tk()
root.title("Skill Based Placement Prediction")
root.configure()



vcmd_10_to_20 = (root.register(validate_input_10_to_20), '%P')  # Validation for 10 < x <= 20
vcmd_0_to_10 = (root.register(validate_input_0_to_10), '%P')



w2 = Label(root, justify="center", text=" Placement Prediction System ", fg="blue")
w2.config(font=("Comic Sans MS", 20))
w2.grid(row=1, column=0, padx=100)

# Labels
input1 = Label(root, text="Quants", font=('Arial', 12), anchor="w", width=30)
input2 = Label(root, text="Logical Reasoning ", font=('Arial', 12), anchor="w", width=30)
input3 = Label(root, text="Verbal", font=('Arial', 12), anchor="w", width=30)
input4 = Label(root, text="Programming", font=('Arial', 12), anchor="w", width=30)
input5 = Label(root, text="CGPA", font=('Arial', 12), anchor="w", width=30)
input6 = Label(root, text="Networking", font=('Arial', 12), anchor="w", width=30)
input7 = Label(root, text="Cloud Computing", font=('Arial', 12), anchor="w", width=30)
input8 = Label(root, text="WebServices", font=('Arial', 12), anchor="w", width=30)
input9 = Label(root, text="DataAnalytics", font=('Arial', 12), anchor="w", width=30)
input10 = Label(root, text="QualityAssurance", font=('Arial', 12), anchor="w", width=30)
input11 = Label(root, text="Artificial Intelligence", font=('Arial', 12), anchor="w", width=30)

input1.grid(row=2, column=0)
input2.grid(row=3, column=0)
input3.grid(row=4, column=0)
input4.grid(row=5, column=0)
input5.grid(row=6, column=0)
input6.grid(row=7, column=0)
input7.grid(row=8, column=0)
input8.grid(row=9, column=0)
input9.grid(row=10, column=0)
input10.grid(row=11, column=0)
input11.grid(row=12, column=0)

# Variables
input1value = StringVar()
input2value = StringVar()
input3value = StringVar()
input4value = StringVar()
input5value = StringVar()
input6value = StringVar()
input7value = StringVar()
input8value = StringVar()
input9value = StringVar()
input10value = StringVar()
input11value = StringVar()

# Entry fields with validation
input1entry = Entry(root, width=50, textvariable=input1value, validate="key", validatecommand=vcmd_10_to_20)
input2entry = Entry(root, width=50, textvariable=input2value, validate="key", validatecommand=vcmd_10_to_20)
input3entry = Entry(root, width=50, textvariable=input3value, validate="key", validatecommand=vcmd_10_to_20)
input4entry = Entry(root, width=50, textvariable=input4value, validate="key", validatecommand=vcmd_10_to_20)
input5entry = Entry(root, width=50, textvariable=input5value, validate="key", validatecommand=vcmd_0_to_10)
input6entry = Entry(root, width=50, textvariable=input6value, validate="key", validatecommand=vcmd_0_to_10)
input7entry = Entry(root, width=50, textvariable=input7value, validate="key", validatecommand=vcmd_0_to_10)
input8entry = Entry(root, width=50, textvariable=input8value, validate="key", validatecommand=vcmd_0_to_10)
input9entry = Entry(root, width=50, textvariable=input9value, validate="key", validatecommand=vcmd_0_to_10)
input10entry = Entry(root, width=50, textvariable=input10value, validate="key", validatecommand=vcmd_0_to_10)
input11entry = Entry(root, width=50, textvariable=input11value, validate="key", validatecommand=vcmd_0_to_10)

input1entry.grid(row=2, column=1)
input2entry.grid(row=3, column=1)
input3entry.grid(row=4, column=1)
input4entry.grid(row=5, column=1)
input5entry.grid(row=6, column=1)
input6entry.grid(row=7, column=1)
input7entry.grid(row=8, column=1)
input8entry.grid(row=9, column=1)
input9entry.grid(row=10, column=1)
input10entry.grid(row=11, column=1)
input11entry.grid(row=12, column=1)

# Buttons
lr = Button(root, text="Predict", height=1, width=15, bg="blue", fg="white", command=message)
lr.config(font=("Comic Sans MS", 15))
lr.grid(row=13, column=1, pady=10)

rs = Button(root, text="Reset Inputs", command=reset, bg="white", fg="purple", width=15)
rs.config(font=("Times", 15, "bold italic"))
rs.grid(row=15, column=1, padx=10)

ex = Button(root, text="Exit System", command=Exit, bg="white", fg="red", width=15)
ex.config(font=("Times", 15, "bold italic"))
ex.grid(row=15, column=0, padx=10, pady=10)


myLabel = Label(root, height=3, width=50, bg="gray94", fg="black", borderwidth=0, font=("Comic Sans MS", 12), text="", anchor="w", justify="left")
myLabel.grid(row=20, column=1)
t3 = Text(root, height=12, width=50, bg="gray94", fg="black", borderwidth=0, font=("Comic Sans MS", 12),)
t3.grid(row=21, column=1, padx=5)

root.mainloop()
