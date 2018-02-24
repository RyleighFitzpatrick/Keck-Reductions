# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 USEFUL FUNCTIONS FOR DATA REDUCTION                   #
#                                                                       #
#                   CREATED BY: M. Ryleigh Fitzpatrick                  #
#                                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import cosmics
from scipy import interpolate, ndimage
from scipy.optimize import minimize
from scipy.ndimage.interpolation import shift


def cosmicRayCleanup(image, outputImage, gain=5.0, readnoise=10.0, sigclip = 5.0, sigfrac = 0.3, objlim = 5.0):
    """Perform cosmic ray cleaning using the Laplacian Edge Detection Method
        This script employs the cosmics.py package written by Pieter G. van Dokkum
        
        INPUTS:
            image --> string of the name of the image on which to perform
                      cleaning
            outputImage --> string of the name of the file to write out the 
                            cleaned image
    """
    # Read the FITS image as 2D numpy array
    img, hdr = cosmics.fromfits(image)

    # Build the object
    c = cosmics.cosmicsimage(img, gain=gain, readnoise=readnoise, sigclip = sigclip, 
                                                                 sigfrac = sigfrac, objlim = objlim)
    # Run the full cosmic ray reduction
    c.run(maxiter = 4)

    # Write the cleaned image into a new FITS file, conserving the original header
    cosmics.tofits(outputImage, c.cleanarray)

def readFits(fileName, head='no'):
    """ Open fits file and header
    
    INPUTS:
        fileName--> fits file to be opened
        head--> 'yes' if header to be included 
                preset to 'no' to ignore header info
    
    OUTPUT:
        Data array and if requested header info array
    """
    
    hdulist = fits.open(fileName, ignore_missing_end=True, output_verify='ignore')
    
    #separate data and header
    data = hdulist[0].data
    header = hdulist[0].header
    
    #Return data or data and header based on function call
    if head == 'yes':
        return data, header
    else:
        return data

    
def plotFitsImage(data, vmin=None, vmax=None, title='', save='no'):
    """Plot fits image

    INPUTS:
        data --> data to plot, written for 2D fits image
        vmin, vmax --> min and max stretching values
        title --> title of plot
        save--> determine whether to save output plot into current directory, default is no
                in order to save image, set save to desired filename of output

    OUTPUT: 
        NO FUCTION RETURNS
        Image plot of data based on input paramaters or saved image of plot in current directory if paramater
        save = 'File_Name'
    """
    #Set stretching min and max to min and max of data divided by the median if no input provided
    if vmin==None:
        vmin = data.min()/np.median(data)
    if vmax==None:
        vmax = data.max()/np.median(data)

    #Plot via Imshow    
    plt.clf()
    plt.imshow(data, origin = 'lower', vmin = vmin, vmax = vmax, cmap='plasma')
    plt.title(title) 
    plt.colorbar() #Add colorbar

    #Save figure as name input in function call
    if save!='no':
        plt.savefig(save)

def plotFitsAndLine(data, slope, yint, vmin=None, vmax=None, title='', save='no'):

    """Plot fits image with lines overlaying it --> designed to check order separation of fits image
   
 INPUTS:
        data --> data to plot, written for 2D fits image
        slope --> array of slope values for each order separation
        yint --> array of y- intercept values for each order separation
        vmin, vmax --> min and max stretching values
        title --> title of plot
        save--> determine whether to save output plot into current directory, default is no
                in order to save image, set save to desired filename of output to save image

        OUTPUT:
            NO FUCTION RETURNS
            Image plot of data based on input paramaters or saved image of plot in current directory if paramater
            save = 'File_Name'
"""
    
    #If stretching not defined in functoin call, calculate based on data vals
    if vmin==None:
        vmin = data.min()/np.median(data)
    if vmax==None:
        vmax = data.max()/np.median(data)

    fig, ax = plt.subplots()
    #Plot fits image via Imshow    
    ax.imshow(data, origin = 'lower', vmin = vmin, vmax = vmax, cmap='plasma')
    ax.set_title(title) 
                    
    i=0
    for m in slope:
        b = yint[i]
        ax.plot([0, data.shape[0]], [b, m*data.shape[0]+b], linestyle = '-', color = 'cyan', linewidth=2.0)
        i=i+1
    
    ax.set_xlim(0, data.shape[0])

    #Save figure as name input in function call
    if save!='no':
        plt.savefig(save)

def plotPairs(pairs,*args,**kwargs):
    """Plot a spectrum of the form [[Lambda, Intensity],
                                    [Lambda, Intensity]
                                            ...       ]"""
    plt.plot(pairs[:,0], pairs[:,1], *args, **kwargs)

def normalize(pairs):
    """Given (x,y) pairs, normalize them to have avg y-value 1."""

    return np.column_stack((pairs[:,0],pairs[:,1]/np.mean(pairs[:,1])))



def fullprint(*args, **kwargs):
  """ Print out entire array, without truncating

  INPUTS:
    Array to be printed in full
    Any args or kwargs accepted by python pprint function for specific output
 
 OUTPUT:
    Full print out of arrayS
    """
  
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold='nan')
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)
        

def getColumn(rawData, colCenter, colWidth):
    """Extract multi pixel column from 2D fits image of echelle spectra and median smooth along spectral
       dispersion direction
    
    **Made to aid in identifying echelle order positons
    
    INPUTS:
        rawData--> 2D fits image data 
        colCenter--> center pixel (in spectral direction) for column (integer)
        colWidth--> width of column in spectral dispersion direction (integer)
    
    OUTPUT:
        median smoothed column, perpendicular to echelle orders for Keck/NIRSPEC-AO Spectrum
    """
    
    #Make sure column width is even
    if colWidth%2!=0:
        colWidth = colWidth-1
        
    #Find start and end pixels of column in spectral dispersion direction
    min = colCenter - colWidth/2
    max = colCenter + colWidth/2
    
    #Extract column from data and median smooth in spectral dispersion direction
    colMedian = rawData[:,min:max] #choose column of data 20 pixels wide
    colMedian = np.median(colMedian,1)
    
    return colMedian    


def boxcarSmooth(data, boxSize, type = 'median'):
    """Smooth a data set with a boxcar method - find the median of data points in a certain boxsize and assign that
    value as the smoothed value for the center pixel, shift box 1 pixel and repeat
    
    INPUTS:
        data--> data array to be smoothed
        boxSize--> size of box to use to take median and perform smoothing
        
    OUTPUT:
        Median smoothed data set and corresponding x coordinates (in pixels)
    
    """
    
    pixelNum = np.arange(boxSize/2,len(data))
    smoothData = np.arange(len(data)-(boxSize/2))
    i=0
    
    #Get Median value in each "box"
    while i<data.shape[0]-boxSize/2:
        #for x in range(i,boxSize+i):
         #   print i
        dataBox = data[i:boxSize+i].copy()  #Take elements i to boxSize+i

        if type == 'median':
            smoothData[i] = np.median(dataBox) #Calculate median and output as smoothed value
        elif type == 'mean':
            smoothData[i] = np.mean(dataBox) #Calculate mean and output as smoothed value
        else:
            print("Invalid smoothing type, please enter median or mean")
        i = i+1
        
    return smoothData, pixelNum

def findJumps(Data):
    """ 
    Calculate the "jumps" or difference between values of an array, taking the of the defference n+1 and n-1 
    elements as each difference value.
    
    INPUTS: 
        Data --> 1D data array on which to perform difference calculations

    OUTPUT:
        Array of the calculated difference values: diff_n = (n+1) - (n-1)
    """
    
    i=1
    difference = Data.copy()
    end = len(Data) #Find stopping point, end of data array
    difference[0]=0 #Set the first difference value to 0 (no n-1 element exists)
    difference[end-1]=0 #Set the last difference value to 0
   
   #Assign the difference value as the difference of the point before and the point after
    while i<end-1:
        difference[i] = Data[i+1]-Data[i-1]
        i=i+1
    return difference

def findPeakCandidates(smoothDiff, smoothDiffX):
    """Identify peak candidates 

    Designed to use the difference of a smoothed function, found with the findJumps function and 
     smoothed with boxcarSmooth function
    
    INPUT:
        smoothDiff --> smoothed function from findJumps, identifying start and end of echelle orders
        
    OUTPUTS:
        peakX --> X values (spatial coord) of peak candidates, corresponds to vertical (Y) spatial axis of fits image
        peak --> Y values of peak candidates   
    """

    i = 0
    dev = []
    devX = []
    #Iterate through smoothDiff and find the difference between consecutive points
    while i < len(smoothDiff)-1:
        dev = np.append(smoothDiff[i+1] - smoothDiff[i], dev)
        #Assign corresponding X value between the two X values
        devX = np.append((smoothDiffX[i+1] + smoothDiffX[i])/2, devX)
        i=i+1

    i=0
    peak = []
    peakX = []
    #Iterate through deviation values of smoothDiff and identify points that have deviation larger than 1.0
    for d in dev:
        if d!=0.0 and d!=1.0:
            p=int(devX[i])
            peak = np.append(smoothDiff[p], peak)
            #Keep track of X coords of candidate peaks chosen
            peakX = np.append(devX[i], peakX)
        i=i+1
        
    return peakX, peak


def findLocalMinMaxPeaks(X, Y, increment):
    """Find local maximum and minimum within specified regions, used for order separation.
    
    INPUTS:
        X --> x values of data points to be evaluated
        Y --> y values of data points to be evaluated
        increment --> array of x values of divisions between regions
        
    OUTPUTS:
        mins / maxs --> minimum and maximum values (values of Y corresponding to mins and max's)
        minXs / maxXs --> values and X corresponding to minimum and maximum values
    """
    
    minIs = []
    maxIs = []
    j = increment[0]
    for endPoint in increment[1:]:    
        i = j # i indicates the left boundary of the interval 
        while (j<len(X) and X[j] < endPoint): # walk j over to the right boundary
            j = j+1
        # get min/max indices for slice (needs to be offset by i)
        minIs.append(i + np.argmin(Y[i:j]))
        maxIs.append(i + np.argmax(Y[i:j]))

    mins = Y[minIs]
    maxs = Y[maxIs]
    minXs = X[minIs]
    maxXs = X[maxIs]
    
    #Add something to make sure it alternates between min and max????
    
    return mins, minXs, maxs, maxXs

def findOrderSeparationFromSpectrum(spectrum, vertSliceWidth=20, plot='no', 
                                    orderSepGuess = [0, 100, 200,320,440,560,700, 840, 960]):
    """Perform order separation for echelle spectra. Defaults set for Keck/NirspecAO with filter 
        Nirspec-5-AO.
        **Based on Spectral Image --> Use function findOrderSeparationFromFlat based on flat 
            field image if possible, it will produce better results so long as the echelle 
            position and x-disp position have not moved between the flat field and image.

    INPUTS:
        spectrum --> echelle spectral image
        vertSliceWidth --> width of the vertical slice to median combine and eliminate
                            noise when finding the order separation
        plot --> Plot the spectrum and calculated lines for the order separation.
                 Defaults to 'no'.
        orderSepGuess --> List of pixel values for the users guess at the echelle order separation. This
                        should provide an approximate order separation by which to arrange the calculation 
                        of the actual separation and indicate the number of echelle orders in the image.

    OUTPUT:
        Slope and Y-int values for the order separation in pixels

    """

    imgSize = spectrum.shape[1]

    #Find order separation vertical coordinates on the left and right hand sides of the detector
    orderSep_L = findOrderSepCoordsForSingleCut(spectrum, orderSepGuess, 100, colWidth=vertSliceWidth)
    orderSep_R = findOrderSepCoordsForSingleCut(spectrum, orderSepGuess, 900, colWidth=vertSliceWidth)

    #Define separation between echelle orders by fitting a line to the two values found on either side 
    #of the CCD for each order
    orderSepSlope, orderSepYInt = calculateLine(orderSep_L, orderSep_R)

    #Shift YInt up by 1/2 of boxcar Width
    orderSepYInt = [x + (vertSliceWidth/2.) for x in orderSepYInt ]

    if plot!='no':
        plotFitsAndLine(spectrum, orderSepSlope, orderSepYInt, title='Order Separation')

    return orderSepSlope, orderSepYInt


def findOrderSepCoordsForSingleCut(spectrum, orderSepGuess, colCenterPixel, colWidth=20):
    """
    Find the order separation coordinates for a single vertical cut along an echelle spectrum fits image.

    INPUTS:
        spectrum --> the fits image of an echelle spectrum
        orderSepGuess --> List of pixel values for the users guess at the echelle order separation. This
                        should provide an approximate order separation by which to arrange the calculation 
                        of the actual separation and indicate the number of echelle orders in the image.
        colCenterPixel --> Spatail direction pixel on which to center the guess. This allows the order
                           separation values to be determied for different vertical cuts.
        colWidth --> pixel width for which to perform median combine to assist in eliminating noise before
                     determining the order separation pixel

    OUTPUT:
        Pixel values for the order separation at the desired vertical cut across the detector

"""

    #Get median combined column/vertical cut (approx. perpendicular to echelle orders)
    colMedian = getColumn(spectrum, colCenterPixel, colWidth) #centered on x=100, 20 pixels wide

    #Boxcar smooth column plot for easier order identification
    boxcar, boxcarX= boxcarSmooth(colMedian, colWidth)

    #Find difference in y values between points of smoothed column plot
    diff = findJumps(boxcar)

    #Smooth the difference to use (next step) to find peaks and identify split between each echelle order
    smoothDiff, smoothDiffX = boxcarSmooth(diff, 3)

    #Identify peak candidates in smoothed difference function of the smoothed column plot
    peakX, peak = findPeakCandidates(smoothDiff, smoothDiffX)

    #Identify local max and min inside regions specified by inc to more accurately define order start/end
    mins, minXs, maxs, maxXs = findLocalMinMaxPeaks(peakX, peak, orderSepGuess)

    #Find order separation coordinate (Y value on fits image spatial axis) by taking value between min (end of one order) 
    #and max (start of next order)
    orderSep = separateMaxMinPairs(minXs, maxXs)

    return orderSep

     #if plot!='no':
    #    fig, ax = plt.subplots()


def findOrderStartEndByVertCut(flatImg,  colCenterPixel, colWidth=20, 
                               orderSepGuess=[0, 50, 160, 270, 360, 480, 600, 740, 870, 1000],
                               plot='no', returnOrderStartEnd='no', leftBoundary='min'):

    """
    Find the order separation coordinates for a single vertical cut along an echelle spectrum fits image.

    INPUTS:
        flatImg--> the fits image of an echelle spectrum flat field
        colCenterPixel --> Spatail direction pixel on which to center the guess. This allows the order
                           separation values to be determied for different vertical cuts.
        colWidth --> pixel width for which to perform median combine to assist in eliminating noise before
                     determining the order separation pixel
        orderSepGuess --> List of pixel values for the users guess at the echelle order separation. This
                        should provide an approximate order separation by which to arrange the calculation 
                        of the actual separation and indicate the number of echelle orders in the image.
        plot -->
        returnOrderStartEnd -->
        leftBoundary --> 


    OUTPUT:
        Pixel values for the order separation at the desired vertical cut across the detector.
        If returnOrderStartEnd = 'yes', then also return list of order (start,end) pixel values as a lsit
        of tuples.
       
    """

    #Get median combined column approx. perpendicular to echelle orders and smooth for easier order identification
    colMedian = getColumn(flatImg, colCenterPixel, colWidth) #centered on colCenterPixel, colWidth pixels wide
    
    #Boxcar smooth column plot for easier order identification
    boxcar, boxcarX= boxcarSmooth(colMedian, 1)

    #Find difference in y values between points of smoothed column plot
    #diff = findJumps(boxcar)
    diff = findJumps(boxcar)

    #Smooth the difference to use (next step) to find peaks and identify split between each echelle order
    #smoothDiff, smoothDiffX = boxcarSmooth(diff, 3)

    #Identify peak candidates in smoothed difference function of the smoothed column plot
    #peakX, peak = findPeakCandidates(smoothDiff, smoothDiffX)
    peakX, peak = findPeakCandidates(diff, boxcarX)

    #Identify local max and min inside regions specified by inc to more accurately define order start/end
    mins, minXs, maxs, maxXs = findLocalMinMaxPeaks(peakX, peak, orderSepGuess)

    #Find order separation coordinate (Y value on fits image spatial axis) by taking value between min 
    #(end of one order) and max (start of next order)
     
    #Find order spearation boundaries
    if leftBoundary is 'min':
        #Take left boundary as min only, and other's as average of min & max
        orderSep = []
        maxXs[0] = minXs[0]
        orderSep.append([(x+y)/2. for x,y in zip(minXs,maxXs)])

    elif leftBoundary is 'average':
        orderSep = [(x+y)/2. for x,y in zip(minXs,maxXs)]

    elif leftBoundary is 'max':
        #Take left boundary as min only, and other's as average of min & max
        orderSep = []
        minXs[0] = maxXs[0]
        orderSep.append([(x+y)/2. for x,y in zip(minXs,maxXs)])
    else:
        print("Please enter min, max, or average for left boundary. This determines\
                how to treat the left order separation. Default is to take the\
                minimum (start) of order as the boundary.")
        return

    #Plot if requested in parameters
    if plot!='no':
        plt.figure(figsize=(20,6)) 
        plt.plot(colMedian)
        #plt.plot(boxcarX, boxcar, color='r')
        plt.plot(diff - 8000, color = 'g')
        plt.plot(minXs, mins - 8000, linestyle=' ', marker = '+', color='r', mew=3, ms=10)
        plt.plot(maxXs, maxs - 8000, linestyle=' ', marker = '+', color='cyan', mew=3, ms=10)
        for sep in orderSep[0]:
            plt.axvline(x=sep, color = 'cyan')


    if returnOrderStartEnd!='no':
        if leftBoundary is 'min':
            maxXs[0] = 0
        elif leftBoundary is 'max':
            minXs[0] = 0
            
        orderStartEnd = zip(maxXs,minXs)

        i=0
        for a,b in orderStartEnd:
            if a>b:
                orderStartEnd[i] = (b,a)
            i+=1
        
        return orderSep, orderStartEnd
    else:
        return orderSep

def findOrderSeparationFromFlatByVertCut(flatImg, startPixel, endPixel, colWidth=20, plot='no'):

    #Define lists for order separation and start/end values
    orderSeparation = []
    orderStartEnd = []

    #Find order start/end value and split pixel for every vertical cut (each 
    #pixel) in the spectral direction
    for pixel in np.arange(startPixel, endPixel, 1):
        orderSepCut, orderStartEndCut = findOrderStartEndByVertCut(flatImg, pixel, colWidth=colWidth, 
                                                                      returnOrderStartEnd='yes')
        #Add order separation and start/end pixel values for each vertical cut
        orderSeparation.append(orderSepCut)
        orderStartEnd.append(orderStartEndCut)

    #convert lists or order separation and start/end pixel values by vertical cuts to 
    #numpy arrays, and return
    orderSeparation = np.array(orderSeparation)
    orderStartEnd = np.array(orderStartEnd)

    if plot != 'no':
        Xs = []
        Reds = []
        Greens = []

        for i in np.arange(0,orderStartEnd.shape[0],1):
            for j in np.arange(0,orderStartEnd.shape[1],1):
                Xs.append(i)
                Reds.append(orderStartEnd[i,j,0])
                Greens.append(orderStartEnd[i,j,1])

        fig, ax = plt.subplots()
        ax.imshow(flatImg, vmin=flatImg.min(), vmax=flatImg.max(), origin='lower')       
        #Plot start of order
        ax.plot(Xs, Reds, marker='o', color='r', linestyle='', ms=3)
        #Plot end of order
        ax.plot(Xs, Greens, marker='o', color='g', linestyle='', ms=3)

    return orderSeparation, orderStartEnd



def findApproxOrderSeparationLineFromFlat(flatImg, orderCenGuess=[0, 50, 160, 270, 360, 480, 600, 740, 870, 1000], 
                                plot='no', LRPlot='no'):
    """Perform order separation for echelle spectra from flat field image. Defaults set for Keck/NirspecAO with filter 
        Nirspec-5-AO.
        **Based on Flat Field Image --> Use function findOrderSeparationFromSpectrum if flat 
            field image is not available, this will produce better results if the echelle 
            position and x-disp position have not moved between the flat field and science image.

    INPUTS:
        spectrum --> echelle spectral image
        vertSliceWidth --> width of the vertical slice to median combine and eliminate
                            noise when finding the order separation
        orderSepGuess --> List of pixel values for the users guess at the echelle order separation. This
                        should provide an approximate order separation by which to arrange the calculation 
                        of the actual separation and indicate the number of echelle orders in the image.
        plot --> Plot the spectrum and calculated lines for the order separation.
                 Defaults to 'no'.
        LRPlot --> Plot the vertical cuts, order start/end values, and order separation found for each
                    side of the detector.

    OUTPUT:
        Slope and Y-int values for the order separation in pixels
        
    """

    #Find order separation vertical coordinates on the left and right hand sides of the detector
    orderSep_L = findOrderStartEndByVertCut(flatImg, 100, orderSepGuess= orderCenGuess,colWidth=20, plot=LRPlot)
    orderSep_R = findOrderStartEndByVertCut(flatImg, 900, orderSepGuess= orderCenGuess,colWidth=20, plot=LRPlot)

    #Define separation between echelle orders by fitting a line to the two values found on either side of the CCD for each order
    orderSepSlope, orderSepYInt = calculateLine(orderSep_L, orderSep_R)

    #Plot the flat image and lines of separation for each order if plot keyword is not 'no'
    if plot!='no':
        plotFitsAndLine(flatImg, orderSepSlope, orderSepYInt, title='Order Separation', vmin=flatImg.min(), vmax=flatImg.max())

    return orderSepSlope, orderSepYInt
  


def separateMaxMinPairs(minX, maxX):
    """Take in arrays with minimum/maximum X coords for echelle orders and find the midpoint between the end of
    one order and the beginning of the next

    INPUTS:
         minX --> X coordinates of minimums (echelle order outer edge)
         manX --> X coordinates of maximums (echelle order inner edge)

    OUTPUT:
         orderSep --> array of the X coord (for spatial axis, so Y coord for fits image) between orders for given input
    """

    orderSep = []

    i=0
    for mn in minX:
        if i< len(minX)-1:
            mx = maxX[i+1]
            orderSep.append((mn+mx)/2.0)
        else:
            orderSep.append((mn+1000)/2.0)
        i=i+1

    return orderSep



def calculateLine(yL, yR, xL = 100., xR = 900.):
    """Calculate line for separation between echelle orders by fitting a line to the two values found on either side 
    of the CCD

    INPUTS:
         yL --> y values for order separation from left side of the CCD 
         xL --> x-coord for which the column was centered to calculate order separation on left side (defaults to 100)
         yR --> y values for order separation from right side of the CCD 
         xR --> x-coord for which the column was centered to calculate order separation on right side (defaults to 900)
    """
    
    slope = []
    yint = []
    
    if len(yL) == len(yR):
        #Iterate through order separation values
        i = 0
        for y1 in yL:
            y2 = yR[i]
            m = (y2 - y1)/(xR - xL)
            slope.append(m)
            b = y1 - m*xL
            yint.append(b)  
            i=i+1
    else:
        print("CANNOT PROCEED")
        print("Number of orders do not match. Check order separation calculation.")
        
    return slope, yint

def splitImageByOrder(img, orders, orderStartEnd, edgeEffects=[5,1005]):
    """
    Split a fits image by order, taking vertical cuts and returning the intensity
    values for each vertical cut between the start/end of each order for a given
    cut.
    
    INPUTS:
    img --> image from which to perform order split
    orders --> array of string names for orders to split 
               the image 
    orderStartEnd --> array of start/end values for each order, by vertical cut
                      output of the function
    edgeEffects --> x-coordinates to start and end image, 
                    to get rid of edge effects
                    Defaults to 5 and 1005
                    
    OUTPUT:
        A dictionary with the key referring to the order number, and for each order
        a list of intensity values along each vertical cut in the order, from the
        start to the end.
        
        Note: The length of each vertical cut is not necessarily the same as the
        other vertical cuts. Spatial rectification will need to be performed on
        these cuts.
                    
    The result of this function will be the input for performSpatialRectification,
    or a single dictionary item with an order key for performSpatialRectificationByOrder.
    """
    
    #Define dictionary for each order and vertical cuts
    imageByOrders = {}
    
    
    #Check sizes of orderStartEnd and orders to ensure they match
    image = img[:,edgeEffects[0]:edgeEffects[1]]
    
    #Iterate through each expected order
    for orderNum in orders:
        
        imageCutForOrder = []
        
        #Convert order numbers to index, starting at 0
        index = max(orders) - orderNum #index from 0 to number of orders
        
        orderStartVals = orderStartEnd[:, index, 1] #array of start pixels for order
        orderEndVals = orderStartEnd[:, index+1, 0] #array of end pixels for order
        
        i=0
        #For each vertical cut, take the order start and end value and find the
        #intensity values at each pixel between
        for start, end in zip(orderStartVals, orderEndVals):
            start = int(start)
            end = int(end)
            
            #Find the intensity values at each pixel in the vertical cut
            imageCutForOrder.append(image[start:end, i])
            i+=1
        
        #Assign the list of intensity values by vertical cuts to the proper order key
        #in the dictionary object
        imageByOrders[orderNum] = imageCutForOrder
        
    return imageByOrders

def performSpatialRectificationByOrder(intensity, pixelHeight = 100.):
    
    """
    Perform spatial rectification for a single order. Re-scale and interpolate
    intensity values for each vertical cut, to straighten/rectify the image in 
    the spatial direction.
    
    INPUTS:
        intensity --> list of lists of intensity values for each vertical cut
                      along a single order
                      Note, lists are of varying lengths depending on width
                      of the order at each vertical cut before spatial
                      rectification
        pixelHeight --> Height for which to scale each order, in pixels
        
    OUTPUT:
        Rectified intensity values, on pixel grid (ie: fits image) for a single
        echelle order
        
        This function is called by performSpatialRectification to spatially
        rectify each order
        
    """
    
    rectifiedIntensity = []
    
    i=0
    #Loop through each vertical cut
    while i< len(intensity): 
        cut = intensity[i] #Pick out vertical cut
        factor = pixelHeight/float(len(cut)) #calculate re-scaling factor
        #Perform intensity value interpolation and re-scaling for vertical cut
        #and append to rectified image
        rectifiedIntensity.append(ndimage.zoom(cut, zoom=factor, order=1))
        i=i+1
        
    #Combine and transpose to have numpy array of intensities where x,y pixels match
    #matrix element i,j
    rectifiedIntensity = np.vstack(rectifiedIntensity).T
        
    return rectifiedIntensity

def performSpatialRectification(imageByOrders, orders):
    """
    Perform spatial rectification for all orders. Re-scale and interpolate
    intensity values for each vertical cut for each order, to straighten/
    rectify the image in the spatial direction.
    
    Calls performSpatialRectificationByOrder for each order
    
    INPUTS:
        imageByOrders --> A dictionary with the key referring to the order 
                          number, and for each order a list of intensity 
                          values along each vertical cut in the order, from 
                          the start to the end.
        orders --> list of integers representing each order, from the bottom
                   of the detector, upwards
                   
    OUTPUT:
        A dictionary with the key referring to the order number, and for each 
        order a rectified 2D array of intensity values for the spectra
    """
    
    #Define dictionary for each order and rectified image
    rectifiedImageByOrders = {}
    
    #Iterate through each order
    for orderNum in orders:
        
        #Perform spatial rectification for the order
        rectifiedImage = performSpatialRectificationByOrder(imageByOrders[orderNum])
        
        #Add to dictionary under order key
        rectifiedImageByOrders[orderNum] = np.vstack(rectifiedImage)
        
    #return spatially rectified fits images for each order
    return rectifiedImageByOrders





#SPECTRAL RECTIFICATION FUNCTIONS

#MODEL RECTIFICATION

def separateByWavelength(spectrum, lambdas, orderSep):
    """
    Pull out spectral region between specified wavelengths. Always returns a 
    spectrum in increasing wavelength order.
    
    INPUTS:
        spectrum --> intensity values } Note: spectrum and intensity arrays MUST
        lambdas --> wavelength values }       be the same size.
        orderSep --> array with [start, end] value for the wavelength
        
    OUTPUTS:
        spectrum between the desired wavelengths, in the form [[wavelength, intensity],
                                                               [wavelength, intensity],
                                                               ....                    ]
    """

    #Pick out order region from model spectrum
    orderRange = np.where((lambdas >= orderSep[0]) & (lambdas <= orderSep[1]))[0]
    
    #Select intensity and wavelength values for desired region
    spec = spectrum[orderRange[0]:orderRange[-1]]
    wavelengths = lambdas[orderRange[0]:orderRange[-1]]

    if wavelengths[0] > wavelengths[-1]:
        #Flip spectrum if necessary
        spec = np.flip(spec, 0)
        wavelengths = np.flip(wavelengths, 0)
    
    return np.column_stack((wavelengths, spec))

def normalizeSpecRegion(spectrum, lambdaRegion = ['full'], minDiff = 0.00001):
    """ Extract and normalize the region of a spectrum between given wavelengths
    
    INPUTS:
        spectrum --> full spectrum to be normalized- 2d array with Lambda, I
        lambdaRegion --> list of length 2 with the start and end wavelengths
        
    OUTPUTS: Normalized spectrum with points between the min and max lamba input
    
    """
    
    wavelength = spectrum[:,0]
    intensity = spectrum[:,1]

    if lambdaRegion == ['full']:
        #Set to full input spectrum
        lambdaRegionIndex = [0]
        lambdaRegionIndex.append(len(wavelength) - 1)
    else:
       #Find the index of the model spectrum corresponding to the lambda values
        lambdaRegionIndex = [np.where(abs(wavelength - lambdaRegion[0]) <= minDiff), 
                             np.where(abs(wavelength - lambdaRegion[1]) <= minDiff)]

        #Convert index array to list
        lambdaRegionIndex = [lambdaRegionIndex[0][0][0], lambdaRegionIndex[1][0][0]]
    
    #Pull out correct region
    newIntensity = intensity[lambdaRegionIndex[0]:lambdaRegionIndex[1]]
    newWavelength = wavelength[lambdaRegionIndex[0]:lambdaRegionIndex[1]]

    #Normalize model spectrum for region between desired wavelengths
    normalizedIntensity = newIntensity/np.mean(newIntensity)
    
    return np.array([list(x) for x in zip(newWavelength, normalizedIntensity)])

def pixelToLambdaSpace(spectrum, lambdaStart, lambdaEnd):
    """Convert from pixel space to approximate wavelength space by 
        dividing into even wavelength increments from given start
        to end value.
        
        INPUTS:
            spectrum --> Region of spectrum to be converted from 
                        pixel space to approximate lambda space.
                        Only include intensity values- do not need
                        pixel coords
            lambdaStart --> minimum lambda value
            lambdaEnd --> maximum lambda value
            
            Note: lambdaEnd > lambdaStart
                This function assumes pixel values increase in
                direction of increasing wavelength. Flip spectrum
                if this is not the case. 
      
      OUTPUTS: 
          spectrum array, with 1st column wavelength values and, 
          second column intensity values. The intensity values
          DO NOT change from initial input.
    """
    
    #add intensity values to new spectrum
    newSpec = np.array(np.vstack(spectrum))[:,0]
    
    #calculate evenly spaced lambda values for each pixel value
    Lambda = np.linspace(lambdaStart, lambdaEnd, num=len(spectrum))
    
    #add lambda values to new spectrum, match with intensity
    newSpec = np.stack((Lambda, newSpec), 1)
    
    #return spectrum with [lambda, intensity]
    return newSpec

def performModelSplineFit(modelStartLambda, modelEndLambda, modelFeatureLambdas, 
                               dataStartX, dataEndX, dataFeatureXs, k=2, 
                               samplesPerPixel=100.):
    """
    Perform a spline fit to match the data X coords of spectral features to the
    correct lambda values, taken from the model spectrum.
    
    INPUTS:
        modelStartLambda --> starting lambda value for the model spectrum
        modelEndLambda --> ending lambda value for the model spectrum
        modelFeatureLambdas --> lambda values of features in the model spectrum,
                                these should match up with the pixel values
                                in dataFeatureXs
        dataStartX --> starting X pixel value for the data spectrum
        dataEndX --> ending X pixel value for the data spectrum
        dataFeatureXs --> pixel values of features in the data spectrum,
                        these should match up with the lambda values
                        in modelFeatureLambdas
        k --> univariate spline fit order, defaulted to 2nd order 
        samplesPerPixel --> sample size per pixel for use in the fit function for
                            resampling, defaulted to 100.
                            
        OUTPUT:
            Lambdas --> array of lambda values on pixel grid for the data spectrum,
                        stretched to fit the data and model, spectrally rectifying
                        the wavelength values. This will be used

    """
    #Define array with the start and end pixels and pixels of the features for which
    #to perform the spline for the science data
    featureX = np.hstack([dataStartX, dataFeatureXs, dataEndX])
    
    #Define array with the start and end wavelengths and those off the features for 
    #which to perform the spline - should match the data spec features in featureX
    modelLambda = np.hstack([modelStartLambda, modelFeatureLambdas, modelEndLambda])
    
    #Fit a interpolated intervariate spline polynomial to the x-coordinate of the 
    #model spectrum, and the actual wavelength of the model spectrum
    spline = interpolate.InterpolatedUnivariateSpline(featureX, modelLambda, k=k)
    
    fitX = np.arange(0, dataEndX, 1./samplesPerPixel)
    fitY = spline(fitX)
    
    #Rescale wavelenth values to match to pixel space of science data
    Lambdas = fitY[np.arange(dataStartX, dataEndX, 1)*int(samplesPerPixel)]
    
    return Lambdas

def performInterpolationResampling(spectrum, Lambdas, specSizePixels=995.):
    """
    Takes output from performModelSplineFit to return the re-sampled wavelength and 
    intensity values of the spectrum, based on the spline fit.
    
    INPUTS:
        spectrum --> Intensity values of the spectrum to be resampled
        Lambdas --> Re-sampled wavelength values for the corresponding spectrum,
                    taken from the output of performModelSplineFit
        specSizePixels --> size of the spectrum, in pixel space
                           defaulted to 995. per Keck Observations
        
    """

    #Perfrom interpolation 
    interpFunct = interpolate.interp1d(Lambdas, spectrum)

    minLambda, maxLambda = Lambdas[0],Lambdas[-1]

    resampledWavelength = np.arange(minLambda,maxLambda,(maxLambda-minLambda)/specSizePixels)

    resampledIntensity = interpFunct(resampledWavelength) 
    
    return resampledWavelength, resampledIntensity

def matchSpecResolution(spec, matchSpec):
    """Resample a spectra to match the resolution of another, or 
    to a particular resolution.
    Ensure start and end wavelengths match between spec and matchSpec.
    
    INPUTS:
        spec --> spectrum to be re-sampled [wavelength, intensity]
        matchSpec --> spectrum to match resolution to, this is None if
                    a desired length is specified
        
    OUTPUT:
            re-sampled spectrum of spec input to match resolution of
            matchspec   
    """
    
    #Calculate interpolation function
    interpFunct = interpolate.interp1d(spec[:,0], spec[:,1]) 
    resampledIntensity = interpFunct(matchSpec[:,0])
    
    #Return resampled spectrum [lambda, intensity]
    return np.column_stack((matchSpec[:,0], resampledIntensity))

def createRollingWindows(array, width):
    """Create an array with rolling windows of a specified size based on
    input array. ie: array = [1,2,3,4,5] and width =3, then:
    outputArray = [[1,1,2], [1,2,3], [2,3,4], [3,4,5], [4,5,5]]
    
    INPUTS:
        array --> array for which to take a rolling window
        width --> width of the window
        
    OUTPUT:
        Array with lists contaning the rollin window, for each element in 
        the original array
        
    NOTE: Windows of endpoints are set to be the value of the endpoint, when
        extending outside the array.
    """

    windowArray = []

    #define window size and radius
    window = [array[0] for x in xrange(width)]
    radius = width/2

    for i in np.arange(0, len(array)):
        #Windows containing first element
        if i < radius:
            window[radius:width] = array[i:i+radius+1]
            windowArray.append(window[:])
        #Subsequent windows
        elif i >= radius and i < (len(array)-radius):
            window[0:width] = array[i-radius:i+radius+1]
            windowArray.append(window[:])
        #Windows containing last element
        else:
            window[0:radius+1] = array[i-radius:i+1]
            window[width-radius:width] = [array[-1] for x in xrange(width-radius-1)]
            windowArray.append(window[:])

    return windowArray

def findApproxSpecShift(dataSpec, modelSpec, ROI, width = 0.00001):
    """Find aproximate shift value to match data and model spectrum, output
    of this function is used as init for L-BFGS-B minimization of the error.

    INPUTS:
        dataSpec --> normalized data spectrum [lambda, intensity]
        modelSpec --> normalized model spectrum [lambda, intensity]
        ROI --> [startLambda, endLambda] region of interest
        width --> Lambda value of shifts

    OUTPUT --> apprixomate shift value in wavelength, used as init for
            L-BFGS-B minimization of error with more accurate shift vals.
        """
    
    shiftError = []
    
    for step in range(-1*len(modelSpec)/2, len(modelSpec)/2):
        shift = step*width
        shifted = shiftRight(dataSpec, shift, order=1)
        #Normalize shifted spectral region
        #normShifted = ru.normalizeSpecRegion(shifted, ROI)
        normShifted = normalize(segment(shifted, ROI[0], ROI[1]))
        error = np.sum((modelSpec[:,1] - normShifted[:,1])**2)
        shiftError.append([shift, error])
    
    
    shiftError = np.array(shiftError)
    minError = np.min(shiftError[:,1])
    minShift = np.where(shiftError[:,1] == minError)
    minShift = int(minShift[0][0])
    return round(shiftError[minShift][0], 6)

def findSpectralShifts(dataSpec, modelSpec, ROI, order=1,width=0.0001, plot='no'):  
    """
    
    INPUTS:
        reference --> model spectrum [lambda, intensity]
        targetSpec --> target spectrum [estimated lambda, intensity]
        ROI --> region of interest on which to focus the shifting, in
                wavelength
    """
    #Match Spectral Resolution between model and data
    dataSpec = matchSpecResolution(dataSpec, modelSpec)
    
    #Normalize data and model spectrum for region between desired wavelengths
    #normRefSpec = ru.normalizeSpecRegion(refSpec, ROI)
    #normDataSpec = normalize(segment(dataSpec, ROI[0], ROI[1]))
    normModelSpec = normalize(segment(modelSpec, ROI[0], ROI[1]))
    
    
    
    def sq_error(shiftVal):
        #Perform shift
        shifted = shiftRight(dataSpec, shiftVal, order=order)
        #Normalize shifted spectral region
        #normShifted = ru.normalizeSpecRegion(shifted, ROI)
        normShifted = normalize(segment(shifted, ROI[0], ROI[1]))
        #Calculate the error, to be minimized
        return np.sum((normModelSpec[:,1] - normShifted[:,1])**2)

    #Find approximate min of error to get init value
    init = findApproxSpecShift(dataSpec, normModelSpec, ROI, width = 0.00001)
    
    result = minimize(sq_error,init,method='L-BFGS-B',bounds=[(init-width,init+width)])
    
    if plot == 'yes':
        fig, ax =plt.subplots(figsize=(20,10))
        plt.plot(normModelSpec[:,0], normModelSpec[:,1], color='grey')
        plt.plot(normModelSpec[:,0], 
                 normalize(segment(shiftRight(dataSpec, result.x, order=order), ROI[0], ROI[1]))[:,1])

    return result.x

def segment(pairs,startX,endX):
    "Returns the segment of the (x,y) pairs with startX <= x <= endX."
    Xs = pairs[:,0]
    startix = np.argmax(Xs >= startX)
    endix   = np.argmax(Xs > endX)
    return pairs[startix:endix,:]

def shiftRight(pairs,t,order=3):
    """Shift the graph of pairs to the right by t.
    
    This is only valid for pairs with equispaced x-values."""
    # t is indices
    Xs = pairs[:,0]
    Ys = pairs[:,1]
    dx = Xs[1] - Xs[0]
    shiftXs = Xs
    shiftYs = shift(Ys,float(t)/dx,order=order,mode='nearest')
    return np.column_stack((shiftXs,shiftYs))

def splitModelByOrder(modelSpec, order, orderBounds):
    
    #Create array of model spec between order boundaries
    model = separateByWavelength(modelSpec[:,1], modelSpec[:,0], orderBounds)
    
    return model

def splitDataSpecToLambda(data, order, medBounds, orderBounds):
    """Split the data spectrum by wavelength and in the spatial direction by pixel- 
        take median value of I's along spatial (vertical) axis if range is given
    
        INPUTS:
            data --> data spactrum, should be output of spatial rectification
            order --> echelle order to be reduced
            medBounds --> spatial axis pixel boundaries to extract spectra
    """
    #Pull out order from spatially rectified image
    if len(medBounds) == 2:
        medSpec = eval('np.median(data['+str(order)+']['+ str(medBounds[0])+':'+ str(medBounds[1])+ '], 0)')
    else:
        medSpec = eval('np.median(data['+str(order)+']' + str(medBounds))
    
    lambdaSpec = pixelToLambdaSpace(medSpec, orderBounds[0], orderBounds[1])
    
    return lambdaSpec

def resToEvenSpacedXs(pairs):
    """Resample along the 0th axis get evenly spaced pairs 
    (ie: even delta Lambda for a spectrum)
    
    INPUT:
        pairs --> 2D list of pairs ie: [[lambda, I], [lambda, I],. ..]
        
    OUTPUT:
        array of resampled values so pairs[:,0] is evenly spaced
    """ 

    startX = min(pairs[:,0])
    endX = max(pairs[:,0])

    interpFunct = interpolate.interp1d(pairs[:,0], pairs[:,1])
    resampledX= np.linspace(startX, endX, len(pairs[:,1]))
    resampledY = interpFunct(resampledX)

    return np.column_stack((resampledX, resampledY))



def findLambda(dataSpec, modelSpec, ROI, plot = 'no'):
    
    #Calculate shift val with minimum error
    shiftVal = findSpectralShifts(dataSpec, modelSpec, ROI, width=0.00005, plot = plot)
    
    #Return model wavelength, and shifted spectral wavelength
    L = np.median(ROI)
    shiftedL = np.float64(L + shiftVal)
    return [L, shiftedL]

def getModelSplineLambdas(dataSpec, modelSpec, numPoints = 15, plot = 'no'):
    
    #Calculate regions of interest
    startL = min(modelSpec[:,0])
    endL = max(modelSpec[:,0])
    
    #Region boundaries
    bounds = np.linspace(startL, endL, numPoints+1)
    
    ROI = []
    for i in np.arange(1,numPoints):
        ROI.append([bounds[i-1], bounds[i]])
      
    #Calculate central model wavelength and corresponding shifted data 
    #wavelength for each ROI
    Lambdas = []
    for roi in ROI:
        roi[0] = round(roi[0], 8)
        roi[1] = round(roi[1], 8)
        Lambdas.append(findLambda(dataSpec, modelSpec, roi, plot=plot))
        
    return np.array(Lambdas)

def splineInterp(lambdas, data):
    #Spline for data lambdas vs model lambdas
    spline = interpolate.InterpolatedUnivariateSpline(lambdas[:,0], lambdas[:,1], k=2)

    startL = min(data[:,0])
    endL = max(data[:,0])

    size = len(data)

    #Evenly spaced lambda values to plut into spline
    fitX = np.linspace(startL, endL, size)
    dataLambdas = spline(fitX)
    
    interpFunct = interpolate.interp1d(dataLambdas, data[:,1])

    #resampledWavelength = np.linspace(startL,endL, size)
    resampledIntensity = interpFunct(dataLambdas) 
    
    return np.column_stack((dataLambdas, resampledIntensity))

def peformSpecRecByOrderAndVertMedComb(data, model, order, orderBounds, medBounds, numPoints = 15, plot = 'no', indivPlot = 'no'):
    
    #Pull out order 49 from Model Spec
    modelSpec = normalize(splitModelByOrder(model, order, orderBounds))
    
    #Pull out order 49 from spatially rectified data and median combine
    #in spatial direction for cleaner spectrum 
    dataSpec = splitDataSpecToLambda(data, 49, medBounds, orderBounds)
    
    #Ensure model lambda's are evenly spaced
    resModel = resToEvenSpacedXs(modelSpec)
    
    #Resample data image to match resolution of model
    resData = matchSpecResolution(dataSpec, resModel)
    
    #Get Spline Resampling
    lambdas = getModelSplineLambdas(resData, resModel, numPoints = numPoints, plot=indivPlot)
    
    #Get interpolation from spline resampling, and normalize
    specRecData = normalize(splineInterp(lambdas, resData))
    
    if plot != 'no':
        fig, ax = plt.subplots(figsize=(20,10))
        ax.plot(modelSpec[:,0], modelSpec[:,1], color = 'grey', linewidth=2)
        ax.plot(specRecData[:,0], specRecData[:,1], color = 'r')

    return specRecData, modelSpec
 


#ARC LINE RECTIFICATION

def identifyArcLineOfHCut(arcIntensity, row, arcSepGuess, plot = 'no'):
    """
    """

    #Find arcline x coord for each horzontal cut
    HCut = arcIntensity[row,:]
    boxcar, boxcarX = boxcarSmooth(HCut, 3)
    diff_1st= findJumps(boxcar)
    diff = findJumps(diff_1st)
    peakX, peak = findPeakCandidates(diff, boxcarX)

    #Identify minimums of the 2nd difference calculation to find the peaks
    mins, minXs, maxs, maxXs= findLocalMinMaxPeaks(peakX, peak, arcSepGuess)

    #Find average, or midpoint of min and max of each arcline to identify the x-coord of the peak
    arcLineXs = minXs + 1 #+1 accounts for half pixel shift in findJumps function

    #Plot visualization if specified in input paramater
    if plot != 'no':
        plt.figure(figsize=(20,6)) 
        plt.plot(HCut)
        plt.plot(boxcarX, boxcar, color='r')
        plt.plot(diff - 2000, color = 'g')
        plt.plot(minXs, mins - 2000, linestyle=' ', marker = '+', color='r', mew=3, ms=10)
        #plt.plot(maxXs, maxs - 2000, linestyle=' ', marker = '+', color='cyan', mew=3, ms=10)

        for X in arcLineXs:
            plt.axvline(x=X, color='cyan')

    #Return x Coords of the arc lines
    return arcLineXs



def fitArcRow(arcLineX, arcLineList, plot='no'):

    #Fit a 2nd order polynomial to the expected arc line wavelength and actual x coord from the arc lamp image
    fitCoeff=np.polyfit(arcLineX, arcLineList, 2)
    samplesPerPixel = 100

    #Create array of X and Y coords of the functoin that fits the above points
    fitX = np.arange(0, 1024, 1./samplesPerPixel) #use more points for a smoother plot
    fitY = np.polyval(fitCoeff, fitX)

    if plot != 'no':
        plt.plot(arcLineX, arcLineList, linestyle = ' ', marker='o')
        plt.plot(fitX, fitY, color='r')

    Lambda = fitY[np.arange(0, 1024, 1)*samplesPerPixel] #rescale wavelength (Y) values to be 1024 pixels 

    return Lambda

def fitArcInterpSpline(arcLineX, arcLineList, k=2, plot='no'):

    #Use polyfit to estimate start and end points in wavelength
    Lambda = fitArcRow(arcLineX, arcLineList)
    start = Lambda[0]
    end = Lambda[-1]
    #print "s %f e %e" % (start,end)
    
    #Check if start and end points included, if not append values
    #if arcLineX[0] != 0:
    arcLineX = np.hstack([0, arcLineX, 1024])
    arcLineList = np.hstack([start, arcLineList, end])
    
    #Fit a interpolated intervariate spline polynomial to the expected arc line wavelength and actual x coord from the arc lamp image
    spl= interpolate.InterpolatedUnivariateSpline(arcLineX, arcLineList, k=k)
    samplesPerPixel = 100

    #Create array of X and Y coords of the functoin that fits the above points
    fitX = np.arange(0, 1024, 1./samplesPerPixel)
    fitY = spl(fitX)

    if plot != 'no':
        plt.plot(arcLineX, arcLineList, linestyle = ' ', marker='o')
        plt.plot(fitX, fitY, color='g')

    Lambda = fitY[np.arange(0, 1024, 1)*samplesPerPixel] #rescale wavelength (Y) values to be 1024 pixels 

    return Lambda



def getArcInterpolationbyRow(Wavelength, Intensity, row, imgLength = 1024., plot = 'no', arcList = []):
    """

    """

    #Get function to perform interpolation of Intensity values to the actual wavelengths (map based on pixel-wavelength relation found in fitArcRow)
    interpFunct = interpolate.interp1d(Wavelength, Intensity[row,:])

    minLambda ,maxLambda = Wavelength[0],Wavelength[-1] #Get min and max (start and end) wavelength of the mapping

    #Re-sample Wavelength and Intensity based on  interpolation function
    resampledWavelength = np.arange(minLambda,maxLambda,(maxLambda-minLambda)/float(imgLength))
    resampledIntensity = interpFunct(resampledWavelength)


    if plot != 'no':
        plt.figure(figsize=(20,6))
        plt.plot(resampledWavelength,resampledIntensity)

        for X in arcList:
            plt.axvline(x=X, color='cyan')

    return resampledWavelength, resampledIntensity



def SpectralRecByRow(arcImage, row, arcLineList, arcSepGuess):

    #Find actual x values of arc lines in image
    arcLineX = identifyArcLineOfHCut(arcImage, row, arcSepGuess)

    #Get wavelengths spaced properly by curve, this is the input for the interpolation function
    #Wavelength = fitArcRow(arcLineX, arcLineList)
    Wavelength = fitArcInterpSpline(arcLineX, arcLineList)

    #Create interpolate function and get resampled wavelengths and intensities
    resampledWavelength, resampledIntensity = getArcInterpolationbyRow(Wavelength, arcImage, row)

    return resampledWavelength, resampledIntensity



def fullSpectralRec(arcImage, arcLineList, arcSepGuess):
    Wavelengths = []
    Intensities = []

    for i in range(1, arcImage.shape[0]):

        Lambda, I = SpectralRecByRow(arcImage, i, arcLineList, arcSepGuess)

        Wavelengths.append(Lambda)
        Intensities.append(I)

    return Wavelengths, Intensities

    
    



#Either average before the slice is taken (weighted average?) then
#OR take quadratic fits for each row and combine for groups of rows... just average coeffs
