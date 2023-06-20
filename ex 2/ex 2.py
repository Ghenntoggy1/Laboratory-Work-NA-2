# Ex 2. Startup Company
import matplotlib.pyplot as plt
import pprint
import matplotlib.patches as mpatches

def LinearInterpolation(l, lstind):
    new = l[lstind - 1][1] + ((lstind + 2 - lstind)/(lstind + 1 - lstind)) * (l[lstind][1] - l[lstind - 1][1])
    return new


data_dict = {}
with open('dataset_2.txt') as f:
    for line in f:
        if 'Date' in line:
            continue
        (key, val) = line.rstrip().split(',')
        if val != 'Nan':
            data_dict[key] = int(val)
        else:
            data_dict[key] = 'Nan'
lst2 = list(data_dict.items())
lst_final = lst2.copy()
Date1, Value1 = zip(*lst2)
last_index = len(lst2) - 1
anot = []
days = []
for i in range(len(lst2)):
    days.append(i+1)
lst = dict(zip(days, Value1))
lst1 = list(lst.items())
Date, Value = zip(*lst1)
Date = list(Date)
Value = list(Value)
interpolated = []
date2 = []
for ind in range(len(lst1)):
    if Value[ind] == 'Nan':
        new_val = LinearInterpolation(lst1, ind-1)
        interpolated.append(int(new_val))
        date2.append(Date1[ind])
        print(Date1[ind], new_val)
for ind in lst2:
    if ind[1] == 'Nan':
        lst2.remove(ind)

Date4, Value4 = zip(*lst2)
j = 0
k = 0
for i in range(len(lst_final)):
    if lst_final[i][1] == 'Nan':
        plt.plot(date2[j], interpolated[j], "-o", color='red')
        j += 1
    else:
        plt.plot(Date4[k], Value4[k], "-o", color='blue')
        k += 1
plt.xlabel("Date (2020.01 - 2020.03)", fontsize=18)
plt.ylabel("Number of Visitors", fontsize=18)
plt.title("Number of Visitors depending on date", fontsize=18)
plt.grid(True)
blue_patch = mpatches.Patch(color='blue', label='Original Data')
red_patch = mpatches.Patch(color='red', label='Interpolated Data')
plt.legend(handles=[blue_patch, red_patch])
plt.show()
