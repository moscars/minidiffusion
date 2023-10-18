import matplotlib.pyplot as plt
rows = open("loss.txt").readlines()

rows = [row.strip() for row in rows]
rows = [row for row in rows if len(row) > len("Epoch: 613232")]
print(len(rows))


#Step 195 of 200 Long: 0.1696452653219063, Current: 0.15234187245368958
longLosses = []
for row in rows:
    vals = row.split()
    longLoss = float(vals[5][:-1])
    longLosses.append(longLoss)

plt.plot(longLosses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Loss over time")
plt.yscale("log")
plt.show()
