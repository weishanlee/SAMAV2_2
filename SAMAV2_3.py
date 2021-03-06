# -*- coding: utf-8 -*-
"""
Created on 
@author: weishan_lee

Sight-seeing order of Macao World Heritage Sites
Case 1: Helicopter-style (Santa Claus and Traveling Salesman Problem)
        The optimal route is found based on the Simulated Annealing and Metropolis Algorithm.
Version 2_2: Change r in Version 1 to rCoor and r, where 
             rCoor refers to the real coordinates of latitude and longitude while
             r is the normalized coordinates for plotting.
Modification in Version 2_3: 1. make use of matplotlib animation
                             2. Add rAnimation as the collection of r for animation
Comments: Path Plot is ok after replacing sites.X with sites.radiansX.
"""
from math import sqrt,exp, sin, cos, atan2, radians
import numpy as np
import random as rand
import matplotlib.animation as animation
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

# Function Definitions

# Function to calculate the magnitude of a vector
def mag(x):
    return sqrt(x[0]**2+x[1]**2)

# Function to calculate the total length of the tour
def distance():
    #%convert latitude and longitude to km
    # Reference: 
        # https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    rEarth = 6371.0 # approximate radius of earth in km
    distance = 0.0

    for i in range(N+1):
        if i == N:
            lat1 = rCoor[N,0] 
            lon1 = rCoor[N,1] 
            lat2 = rCoor[0,0] 
            lon2 = rCoor[0,1] 
        else:
            lat1 = rCoor[i,0] 
            lon1 = rCoor[i,1] 
            lat2 = rCoor[i+1,0] 
            lon2 = rCoor[i+1,1] 

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance += c
    distance = distance * rEarth
    return distance

# output of the score (distance vs time steps)
def outPutScrVSTime(tRecord, scoreRecord):
    data = {'tRecord': tRecord,'scoreRecord':scoreRecord}
    dfCSV = pd.DataFrame(data)
    dfCSV_file = open('./scoreVSTime.csv','w',newline='') 
    dfCSV.to_csv(dfCSV_file, sep=',', encoding='utf-8',index=False)
    dfCSV_file.close()
    
def outPutSitesOrder(rr):
    ## Write rCoor back to cities datafram

    sitesOrder = []
    
    for i in range(N):
        sitesOrder += [ rr[i,2] ]
     
    sites["sitesOrder"] = sitesOrder
    
    sitesOrder = pd.DataFrame(columns = ['sitesId', 'Name'])
    sitesOrder_file = open("./sightSeeingOrder.csv",'w',newline='') 

    for i in range(N+1):
        if i == N:
            sitesOrder.loc[i] = np.uint32(sites.loc[0].sitesOrder), sites.loc[sites.loc[0].sitesOrder].Name
        else:
            sitesOrder.loc[i] = np.uint32(sites.loc[i].sitesOrder), sites.loc[sites.loc[i].sitesOrder].Name

    sitesOrder.to_csv(sitesOrder_file, sep=',', encoding='utf-8', index=False) 
    sitesOrder_file.close()

def plotRoute(rr, sites):
    x = []
    y = []
    n = [int(num) for num in rCoor[:,2].tolist()]

    for i in range(N+1):
        if i == N:
            x.append( sites.loc[n[0]].X )
            y.append( sites.loc[n[0]].Y )
        else:
            x.append( sites.loc[n[i]].X )
            y.append( sites.loc[n[i]].Y )
    fig, ax = plt.subplots()
    ax.title.set_text("Optimal Tour Path")

    ax.plot(x,y,'k-')
    ax.scatter(x[0],y[0],c='blue')
    ax.scatter(x[1:-1],y[1:-1],c='red')

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    #ax.set_xlabel("Longitude",size = 12)
    #ax.set_ylabel("Latitude",size = 12)
    ax.ticklabel_format(useOffset=False)
    plt.grid(True)
    plt.savefig("optimalTourPath.eps")     
            
# Load world heritage sites locations
sites = pd.read_csv("./macauWHSLoc.csv")
N = sites.shape[0]

# plot coordinates of sites
x = sites.X.tolist()
y = sites.Y.tolist()
n = sites.SiteId.tolist()

fig, ax = plt.subplots()
ax.title.set_text("Coordinates of the Sites")

ax.scatter(x,y)

for i, txt in enumerate(n):
    ax.annotate(txt, (x[i], y[i]))

#ax.set_xlabel("Longitude",size = 12)
#ax.set_ylabel("Latitude",size = 12)
ax.ticklabel_format(useOffset=False)
plt.grid(True)
plt.savefig("coordinatesOfSites.eps")

## add radians of X and Y
sites["radiansX"] = sites.X.apply(radians)
sites["radiansY"] = sites.Y.apply(radians)

## normalize data

sites['normX'] = min_max_scaler.fit_transform(sites.X.values.reshape(-1, 1))
sites['normY'] = min_max_scaler.fit_transform(sites.Y.values.reshape(-1, 1))

# Set up the initial configuration
randomList = rand.sample(range(0, N), N)

## Change sites dataframe to rCoor array
rCoor = np.empty([N+1,3])
for i in range(N):
    j = randomList[i]
    rCoor[i,0] = sites.radiansX[j]
    rCoor[i,1] = sites.radiansY[j]
    rCoor[i,2] = sites.SiteId[j]
    
# Add one more ending site which is identical the starting site
rCoor[N,0] = rCoor[0,0]
rCoor[N,1] = rCoor[0,1]
rCoor[N,2] = rCoor[0,2]

## Change sites dataframe to r array
r = np.empty([N+1,3])
for i in range(N):
    j = randomList[i]
    r[i,0] = sites.normX[j]
    r[i,1] = sites.normY[j]
    r[i,2] = sites.SiteId[j]
    
# Add one more ending site which is identical the starting site
r[N,0] = r[0,0]
r[N,1] = r[0,1]
r[N,2] = r[0,2]

#Calculate the initial distance

score = distance()
initScore = score
minScore = initScore
print("Initial score = {:.5f}\n".format(initScore))

animationOption = True  # False
# Create a figure window
if animationOption == True:
    
    # Add rAnimation as the collection of r for animation
    rAnimation = []
    rAnimation.append(r[:,0:2].tolist()[:])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
    ax.grid()

    line, = ax.plot([], [], 'k-',lw=2)
    dot, = ax.plot([], [], color='red', marker='o', markersize=4, markeredgecolor='red', linestyle='')
    timeStep_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    score_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    
    def init():
        """initialize animation"""
        line.set_data([], [])
        dot.set_data([],[])
        timeStep_text.set_text('')
        score_text.set_text('')
        return line, dot, timeStep_text, score_text

# Simulated annealing
# Main loop
Tmax = 1.0
Tmin = 1e-2
tau = 1e3
targetScore = 4.3

tRecord = []
scoreRecord = []

t0=0 # setting up the beginning of the time "lump"
tRecord += [0]
scoreRecord += [score]

firstInitial = True

while (score>targetScore):
    
    if firstInitial == False: 
        # Set up another initial configuration
        randomList = rand.sample(range(0, N), N)

        ## Change sites dataframe to rCoor array
        rCoor = np.empty([N+1,4])
        for i in range(N):
            j = randomList[i]
            rCoor[i,0] = sites.radiansX[j]
            rCoor[i,1] = sites.radiansX[j]
            rCoor[i,2] = sites.SiteId[j]
    
        # Add one more ending site which is identical the starting site
        rCoor[N,0] = rCoor[0,0]
        rCoor[N,1] = rCoor[0,1]
        rCoor[N,2] = rCoor[0,2]

        ## Change sites dataframe to r array
        r = np.empty([N+1,3])
        for i in range(N):
            j = randomList[i]
            r[i,0] = sites.normX[j]
            r[i,1] = sites.normY[j]
            r[i,2] = sites.SiteId[j]
    
        # Add one more ending site which is identical the starting site
        r[N,0] = r[0,0]
        r[N,1] = r[0,1]
        r[N,2] = r[0,2]
        
        #Calculate the initial distance
        score = distance()

    T = Tmax
    t = 0
    while (T>Tmin):
        # Cooling
        t += 1
        T = Tmax*exp(-t/tau)

        # Choose two sites to swap and make sure they are distinct
        i,j = rand.randrange(1,N),rand.randrange(1,N)
        while i==j:
            i,j = rand.randrange(1,N),rand.randrange(1,N)
                
        # Swap them and calculate the change in score
        oldScore = score
    
        rCoor[i,0],rCoor[j,0] = rCoor[j,0],rCoor[i,0]
        rCoor[i,1],rCoor[j,1] = rCoor[j,1],rCoor[i,1]
        rCoor[i,2],rCoor[j,2] = rCoor[j,2],rCoor[i,2]
    
        r[i,0],r[j,0] = r[j,0],r[i,0]
        r[i,1],r[j,1] = r[j,1],r[i,1]
        r[i,2],r[j,2] = r[j,2],r[i,2]
        
        score = distance()        
        deltaScore = score - oldScore

        try:
            ans = np.exp(-deltaScore/T)
        except OverflowError:
            if -deltaScore/T > 0:
                ans = float('inf')
            else:
                ans = 0.0
    
        # If the move is rejected, swap them back again
        if rand.random() > ans:
            
            rCoor[i,0],rCoor[j,0] = rCoor[j,0],rCoor[i,0]
            rCoor[i,1],rCoor[j,1] = rCoor[j,1],rCoor[i,1]
            rCoor[i,2],rCoor[j,2] = rCoor[j,2],rCoor[i,2]
            
            r[i,0],r[j,0] = r[j,0],r[i,0]
            r[i,1],r[j,1] = r[j,1],r[i,1]
            r[i,2],r[j,2] = r[j,2],r[i,2]
            
            score = oldScore
            if np.abs(score - distance())>1e-5:
                print("score: {}".format(score))
                print("distance: {}".format(distance()))
                print("Error Line 315")

        if animationOption == True:
            rAnimation.append(r[:,0:2].tolist()[:])
                    
        if t%1==0:
            tRecord += [t0+t]
            scoreRecord += [score]
            
        if score < minScore: 
            minScore = score
            outPutScrVSTime(tRecord, scoreRecord)
            outPutSitesOrder(rCoor)
            dt = datetime.now()
            print(dt.year, '/', dt.month, '/', dt.day, ' ',
                  dt.hour, ':', dt.minute, ':', dt.second)
            print("Delta score = {:.5f}".format(deltaScore))
            print("New score = {:.5f}\n".format(score))
        
    t0 = t0 + t # go to next time "lump"
    firstInitial = False
# End of Main Loop
def animate(i):
    """perform animation step"""
    xx = []
    yy = []
    for j in range(N+1):
        xx.append(rAnimation[i][j][0])
        yy.append(rAnimation[i][j][1])
    line.set_data(xx,yy)
    dot.set_data(xx,yy)
    timeStep_text.set_text('time step = %.1f ' % tRecord[i] )
    score_text.set_text('score = %.3f ' % scoreRecord[i] )
    return line, dot, timeStep_text, score_text
    
ani = animation.FuncAnimation(fig, animate, frames=tRecord,
                              interval=1, blit=True, init_func=init)

saveAnimation = False
if saveAnimation == True:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=-1)
    ani.save('path.mp4', writer=writer)

plt.show()

print("The initial total traveling distance = {:.5f} km".format(initScore))
print("The optimal total traveling distance = {:.5f} km".format(score))

# plot score vs t
plt.figure()
plt.title("Traveling Distance vs Iteration")
ax = plt.gca()
scoreVsTime = pd.read_csv( "./scoreVSTime.csv") 
plt.plot(scoreVsTime.tRecord,scoreVsTime.scoreRecord,'k-')
plt.minorticks_on()
minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
minorLocatorY = AutoMinorLocator(5)
ax.set_xlabel("Iteration",size = 16)
ax.set_ylabel("Total Traveling Distance (km)",size = 16)
ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
plt.grid(True)
plt.savefig("distanceVsIteration.eps")
plt.show()   

scoreCheck = distance()
print("The checked optimal total traveling distance = {:.5f} km".format(scoreCheck))

plotRoute(rCoor, sites)
#%% Draw routes when sightseeing order is already saved in sightSeeingOrder.csv file
"""from math import sqrt,exp, sin, cos, atan2, radians
import numpy as np
import random as rand
#from vpython import * 
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

sites = pd.read_csv("./macauWHSLoc.csv")
rCoor = pd.read_csv("./sightSeeingOrder.csv")
N=25

def plotRoute(rr, sites):
    x = []
    y = []
    n = [int(num) for num in rCoor.sitesId.tolist()]

    for i in range(N+1):
        if i == N:
            x.append( sites.loc[n[0]].X )
            y.append( sites.loc[n[0]].Y )
        else:
            x.append( sites.loc[n[i]].X )
            y.append( sites.loc[n[i]].Y )
    fig, ax = plt.subplots()
    ax.title.set_text("Optimal Tour Path")

    ax.plot(x,y,'k-')
    ax.scatter(x[0],y[0],c='blue')
    ax.scatter(x[1:-1],y[1:-1],c='red')

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    #ax.set_xlabel("Latitude",size = 12)
    #ax.set_ylabel("Longitude",size = 12)
    ax.ticklabel_format(useOffset=False)
    plt.grid(True)
    plt.savefig("optimalTourPath.eps")     
    
plotRoute(rCoor, sites)"""