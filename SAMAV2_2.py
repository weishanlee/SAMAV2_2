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
Modification in Version 2_2: 1. distance directly used in km
                             2. the animation update is modified
                             3. Add funcion definition plotRout
Comments: Path Plot is ok after replacing sites.X with sites.radiansX.
"""
from math import sqrt,exp, sin, cos, atan2, radians
import numpy as np
import random as rand
from vpython import * 
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
    data = {'timeStep': tRecord,'score':scoreRecord}
    dfCSV = pd.DataFrame(data)
    dfCSV_file = open('./scoreVSTime.csv','w',newline='') 
    dfCSV.to_csv(dfCSV_file, sep=',', encoding='utf-8',index=False)
    dfCSV_file.close()
    
def outPutSitesOrder(rr):
    ## Write rCoor back to cities datafram

    sitesOrder = []
    
    for i in range(N):
        sitesOrder += [ rr[i,3] ]
     
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
    n = [int(num) for num in rCoor[:,3].tolist()]

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
R = 0.02
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
rCoor = np.empty([N+1,4])
for i in range(N):
    j = randomList[i]
    rCoor[i,0] = sites.radiansX[j]
    rCoor[i,1] = sites.radiansY[j]
    rCoor[i,2] = 0.0
    rCoor[i,3] = sites.SiteId[j]
    
# Add one more ending site which is identical the starting site
rCoor[N,0] = rCoor[0,0]
rCoor[N,1] = rCoor[0,1]
rCoor[N,2] = rCoor[0,2]
rCoor[N,3] = rCoor[0,3]

## Change sites dataframe to r array
r = np.empty([N+1,4])
for i in range(N):
    j = randomList[i]
    r[i,0] = sites.normX[j]
    r[i,1] = sites.normY[j]
    r[i,2] = 0.0
    r[i,3] = sites.SiteId[j]
    
# Add one more ending site which is identical the starting site
r[N,0] = r[0,0]
r[N,1] = r[0,1]
r[N,2] = r[0,2]
r[N,3] = r[0,3]

#Calculate the initial distance

score = distance()
initScore = score
minScore = initScore
print("Initial score = {:.5f}\n".format(initScore))

animation = False  # False
# Set up the graphics
if animation == True:
    cv = canvas(center=vector(0.5,0.5,0.0), background = color.white)
    cv.title=" Iteration = {}.\t Total distance = {:.5f} km".format(0, score)
    for i in range(N):
        if i == 0:
            sphere(pos=vector(r[i,0],r[i,1],0.0),radius=R,color = color.blue)
        else:
            sphere(pos=vector(r[i,0],r[i,1],0.0),radius=R,color = color.red)
    l = curve(pos=r.tolist(),radius=R/4,color = color.black)

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
            rCoor[i,2] = 0.0
            rCoor[i,3] = sites.SiteId[j]
    
        # Add one more ending site which is identical the starting site
        rCoor[N,0] = rCoor[0,0]
        rCoor[N,1] = rCoor[0,1]
        rCoor[N,2] = rCoor[0,2]
        rCoor[N,3] = rCoor[0,3]

        ## Change sites dataframe to r array
        r = np.empty([N+1,4])
        for i in range(N):
            j = randomList[i]
            r[i,0] = sites.normX[j]
            r[i,1] = sites.normY[j]
            r[i,2] = 0.0
            r[i,3] = sites.SiteId[j]
    
        # Add one more ending site which is identical the starting site
        r[N,0] = r[0,0]
        r[N,1] = r[0,1]
        #r[N,2] = r[0,2]
        r[N,3] = r[0,3]
        
        #Calculate the initial distance
        score = distance()
        
        if animation == True:
            # Set up the graphics
            #cv.delete()
            #cv = canvas(center=vector(0.5,0.5,0.0), background = color.white)
            # Note: resetting canvas every time would make the title appear with 
            #       a new line on the browser.
            cv.title=" Iteration = {}.   Total distance = {:.5f} km".format(t0, score)
            l.visible = False
            del l
            for i in range(N):
                if i == 0:
                    sphere(pos=vector(r[i,0],r[i,1],0.0),radius=R,color = color.blue)
                else:
                    sphere(pos=vector(r[i,0],r[i,1],0.0),radius=R,color = color.red)
            l = curve(pos=r.tolist(),radius=R/4,color = color.black)

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
        #rCoor[i,2],rCoor[j,2] = rCoor[j,2],rCoor[i,2]
        rCoor[i,3],rCoor[j,3] = rCoor[j,3],rCoor[i,3]
    
        r[i,0],r[j,0] = r[j,0],r[i,0]
        r[i,1],r[j,1] = r[j,1],r[i,1]
        #r[i,2],r[j,2] = r[j,2],r[i,2]
        r[i,3],r[j,3] = r[j,3],r[i,3]
        
        score = distance()        
        deltaScore = score - oldScore
        #print("deltaScore = {:.5f}".format(deltaScore))

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
            #rCoor[i,2],rCoor[j,2] = rCoor[j,2],rCoor[i,2]
            rCoor[i,3],rCoor[j,3] = rCoor[j,3],rCoor[i,3]
            
            r[i,0],r[j,0] = r[j,0],r[i,0]
            r[i,1],r[j,1] = r[j,1],r[i,1]
            #r[i,2],r[j,2] = r[j,2],r[i,2]
            r[i,3],r[j,3] = r[j,3],r[i,3]
            
            score = oldScore
            if np.abs(score - distance())>1e-5:
                print("score: {}".format(score))
                print("distance: {}".format(distance()))
                print("Error Line 290")
        
        if score < minScore: 
            minScore = score
            outPutScrVSTime(tRecord, scoreRecord)
            outPutSitesOrder(rCoor)
            dt = datetime.now()
            print(dt.year, '/', dt.month, '/', dt.day, ' ',
                  dt.hour, ':', dt.minute, ':', dt.second)
            print("Delta score = {:.5f}".format(deltaScore))
            print("New score = {:.5f}\n".format(score))

        if animation == True:    
            # Update the visualization every 100 moves
            cv.title=" Iteration = {}.   Total distance = {:.5f} km".format(t0+t, score)
            if t%100==0:
                rate(100)
                for i in range(N+1):
                    pos = vector(r[i,0],r[i,1],0.0)
                    l.modify(i,pos)
                    
        if t%10==0:
            tRecord += [t0+t]
            scoreRecord += [score]
        
    t0 = t0 + t # go to next time "lump"
    firstInitial = False
# End of Main Loop

print("The initial total traveling distance = {:.5f} km".format(initScore))
print("The optimal total traveling distance = {:.5f} km".format(score))

# plot score vs t
plt.figure()
plt.title("Traveling Distance vs Iteration")
ax = plt.gca()
enVsTime = pd.read_csv( "./scoreVSTime.csv") 
plt.plot(enVsTime.timeStep,enVsTime.score,'k-')
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
from math import sqrt,exp, sin, cos, atan2, radians
import numpy as np
import random as rand
from vpython import * 
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
    
plotRoute(rCoor, sites)