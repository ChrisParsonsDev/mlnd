#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("./datasets/meta/enron_dataset.pkl", "r"))
#start and exit prints for formatting
print

#Size of dataset
print('Size of Dataset: ' + str(len(enron_data)))
#number of features
print('Number of Features: ' + str(len(enron_data.values()[0])))

#Count of people with particular feature.
count = 0
for user in enron_data:
    if enron_data[user]['poi'] == True:
        count+=1
print
print('Count of people with POI Feature: ' + str(count))

#Prints all the names so we know how to query dataset
if False:
    for name in enron_data.keys():
        print name


#James Prentice's TSV
print
print('James Prentice\'s TSV: ' + str(enron_data['PRENTICE JAMES']['total_stock_value']))

#How many emails from Wesley Cowell to a PoI
print('Emails from Wesley Cowell to a PoI: ' + str(enron_data['COLWELL WESLEY']['from_this_person_to_poi']))

#Stock options of Jeffery K SKILLING
print('Stock Options of Jefferey Skilling: ' + str(enron_data['SKILLING JEFFREY K']['exercised_stock_options']))

#total_payments of Lay, Skilling & Fastow
print
print('Total Paymements of Lay, Skilling & Fastow:')
print('Lay - '+ str(enron_data['LAY KENNETH L']['total_payments']))
print('Skilling - '+ str(enron_data['SKILLING JEFFREY K']['total_payments']))
print('Fastow - '+ str(enron_data['FASTOW ANDREW S']['total_payments']))


#Count of people with Salary feature.
count = 0
for user in enron_data:
    if enron_data[user]['salary'] != 'NaN':
        count+=1
print
print('Count of people with Salary Feature: ' + str(count))

#Count of people with Email feature.
count = 0
for user in enron_data:
    if enron_data[user]['email_address'] != 'NaN':
        count+=1
print
print('Count of people with Email Feature: ' + str(count))

#Percentage with NaN total_payments
print
total = 0
count = 0
for user in enron_data:
    if enron_data[user]['total_payments'] == 'NaN':
        count+= 1
    total += 1
percentage = float((float(count)/float(total))*100)
print('Percentage NaN total_payments: '+str(percentage))

#Percentage POI with NaN total_payments
print
total = 0
count = 0
for user in enron_data:
    if enron_data[user]['poi']:
        if enron_data[user]['total_payments'] == 'NaN':
            count+= 1
        total += 1
percentage = float((float(count)/float(total))*100)
print('Percentage NaN total_payments: '+str(percentage))

#Exit Print
print
