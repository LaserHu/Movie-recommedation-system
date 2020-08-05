import matplotlib.pyplot as plt
X=[5, 10, 20, 40 , 60, 80, 160]
#recall
#userCF=[0.096,0.1096,0.1197,0.1229, 0.1231,0.1226,0.1186]
#itemCF=[0.0955, 0.1094, 0.1182, 0.122719, 0.122705, 0.1234, 0.1189]

#coverage
#userCF=[0.44475,0.354088,0.283433,0.220087, 0.196806,0.177856,0.1331]
#itemCF=[0.4429, 0.35597, 0.280596, 0.218392, 0.174278, 0.194331, 0.130317]

#popularity
userCF=[6.9783, 7.105, 7.205, 7.290, 7.336, 7.367, 7.442]
itemCF=[6.979, 7.107, 7.207, 7.289, 7.364, 7.334, 7.438]
line1, = plt.plot(X, userCF,'ro-',label="userCF ", linestyle='--')
line2, = plt.plot(X, itemCF,'g*:', label="itemCF ", linewidth=5)

# Create a legend for the first line.
first_legend = plt.legend(handles=[line1], loc=1)

# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)
plt.title('Popularity')
plt.xlabel('K value')
plt.ylabel('Popularity Index')
# Create another legend for the second line.
plt.legend(handles=[line2], loc=4)
plt.savefig('./test2.png')
plt.show()
