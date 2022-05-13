#!/usr/bin python
# Written by Francesco Ragone 
# adapted by George Miloshevich
  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
from netCDF4 import Dataset
import subprocess

# input

#name convention and folders of CESM, change it to use other datasets
modelfolder='Data_CESM' # OPERATIONS DIRECTLY FROM DISTONET ARE NOT RECOMMENDED
sampling='daily'
experiment='CAM4_F2000_p144_ctrl'
observable='Z3.500hPa'
varname='Z3'
#string datafolder : modelfolder+'/'+sampling+'/'+experiment+'_'+'batch_'+str(batch).zfill(4)+'.'+observable+'.'+sampling
#string datafile   : experiment+'_'+'batch_'+str(batch).zfill(4)+'.'+observable+'.'+sampling+'.'+str(year).zfill(4)+'.nc'

#number of days in the summer (here I'm using only 90 days for simplicity instead of 92)
ntime=92 #SET TO 92
#starting date of the summer (this should be June 1st, maybe check on the calendar if I made a mistake!)
starttime=150 #INDEED
#last day of the summer
endtime=starttime+ntime
#number of longitudes (set to the value of CESM)
nlon=288

#number of batches
nbatchs=10 #10 #100
#number of years
nyears=100  #100
#number of trajectories (a parameter left from the fact that I wrote these scripts for the rare event algorithm, ignore it and just put it equatl to nyears)
ntrajs=100  #100


#boundaries of the latitude band over which the meridional average is taken
latnorth=int(sys.argv[3])#80#60#70#75#80
latsouth=int(sys.argv[2])#70#50#60#55#40

#threshold=4.5
threshold=float(sys.argv[1])#3.5 #only select years with heatwaves above this threshold
#threshold=-np.inf #take all heatwaves
area = 'Scandinavia'
T = 30 # running mean in days
A_max = np.load('Postproc/A_max_'+area+'_'+str(T)+'.npy') # The set of heatwave extrema each year
#output
outfolder='Hayashi'
outfile='test_Hayashi_output'+area+'_'+str(T)+'_'+str(threshold)+'_'+str(latsouth)+'_'+str(latnorth)


#number of heatwaves that will be analysed given the threshold you will put. you have to know it in advance
sizeheat=np.sum(np.array(A_max)>threshold) # compute the number of heatwaves above the threshold
#sizeheat=100
###############################################################

xfft=np.empty((ntime,nlon),dtype=complex)
fftspec=np.empty((ntime,nlon),dtype=complex)

powercos=np.empty((ntime,nlon,sizeheat),dtype=complex)
powersin=np.empty((ntime,nlon,sizeheat),dtype=complex)
cospec=np.empty((ntime,nlon,sizeheat),dtype=complex)
quadr=np.empty((ntime,nlon,sizeheat),dtype=complex)

counter=0

#here it's 365 because the data are daily
zg_clim=np.zeros((365,nlon))

#here I compute the climatological average
for batch in range(1,nbatchs+1):
    #folder of the data
    datafolder=modelfolder+'/'+experiment+'_'+'batch_'+str(batch).zfill(4)+'.'+sampling+'.'+observable
    for year in range(1,nyears+1):
        traj=year
        #name of the file
        datafile=experiment+'_'+'batch_'+str(batch).zfill(4)+'.'+sampling+'.'+observable+'.'+str(year).zfill(4)+'.nc'
    
        tempfile=str(batch)+'_'+str(year)+'_temp.nc'
        if not os.path.exists(outfolder+'/'+tempfile):
            subprocess.call('cdo -mermean -sellonlatbox,0,360,'+str(latsouth)+','+str(latnorth)+' -selvar,'+varname+' '+datafolder+'/'+datafile+' '+outfolder+'/'+tempfile, shell=True)
        dataset=Dataset(outfolder+'/'+tempfile)
        print("shape = ",dataset.variables[varname].shape)       
        zg=np.squeeze(dataset.variables[varname][:]) # I THINK THIS REMOVES EXTRA AXIS IN SHAPE[0] THAT ZG HAS
        print("shape = ",zg.shape)
        zg_clim[:,:]=zg_clim[:,:]+zg[:,:]  # THIS LIKELY COMPUTES CLIMATOLOGY

zg_clim[:,:]=zg_clim[:,:]/nyears/nbatchs

#here I compute the spectrum. The selection condition will go in these loops
for batch in range(1,nbatchs+1):
    #folder of the data
    datafolder=modelfolder+'/'+experiment+'_'+'batch_'+str(batch).zfill(4)+'.'+sampling+'.'+observable
    for year in range(1,nyears+1):
        if A_max[year-1+100*(batch-1)] > threshold:
            print("Including batch: ", batch, ", year: ",year,", A_max: ", A_max[year-1+100*(batch-1)])
            traj=year
            #name of the file
            datafile=experiment+'_'+'batch_'+str(batch).zfill(4)+'.'+sampling+'.'+observable+'.'+str(year).zfill(4)+'.nc'

            tempfile=str(batch)+'_'+str(year)+'_temp.nc'
            if not os.path.exists(outfolder+'/'+tempfile):
                subprocess.call('cdo -mermean -sellonlatbox,0,360,'+str(latsouth)+','+str(latnorth)+' -selvar,'+varname+' '+datafolder+'/'+datafile+' '+outfolder+'/'+tempfile, shell=True)
            dataset=Dataset(outfolder+'/'+tempfile)
            zg=np.squeeze(dataset.variables[varname][:])-zg_clim

            for t in range(starttime,endtime):
                xfft[t-starttime,:]=np.fft.fft(zg[t,:])/nlon
            cosxfft=2*xfft.real
            sinxfft=2*xfft.imag
            for ilon in range(xfft.shape[1]):
                powercos[:,ilon,counter]=np.fft.fft(cosxfft[:,ilon])*np.conj(np.fft.fft(cosxfft[:,ilon]))/(ntime*ntime)
                powersin[:,ilon,counter]=np.fft.fft(sinxfft[:,ilon])*np.conj(np.fft.fft(sinxfft[:,ilon]))/(ntime*ntime)
                fftspec[:,ilon]=np.fft.fft(cosxfft[:,ilon])*np.conj(np.fft.fft(sinxfft[:,ilon]))/(ntime*ntime)
            cospec[:,:,counter]=fftspec.real
            quadr[:,:,counter]=-fftspec.imag
            counter=counter+1
print("counter = ",counter)

plt.figure(figsize=(6, 6))
plt.contourf(np.fft.fftshift(cosxfft,axes=1), cmap='seismic', levels=np.linspace(-np.max(cosxfft), np.max(cosxfft), 18), extend='both')
plt.colorbar()
plt.show()

plt.figure(figsize=(6, 6))
plt.contourf(zg, cmap='seismic', levels=np.linspace(-np.max(zg), np.max(zg), 18), extend='both')
plt.colorbar()
plt.show()


powercos_1s=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)
powersin_1s=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)
cospec_1s=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)
quadr_1s=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)

powercos_1s[:,:,:]=powercos[0:int(ntime/2),0:int(nlon/2),:];
powercos_1s[1:int(ntime/2)-1,1:int(nlon/2)-1,:]=2*powercos_1s[1:int(ntime/2)-1,1:int(nlon/2)-1,:]
powersin_1s[:,:,:]=powersin[0:int(ntime/2),0:int(nlon/2),:];
powersin_1s[1:int(ntime/2)-1,1:int(nlon/2)-1,:]=2*powersin_1s[1:int(ntime/2)-1,1:int(nlon/2)-1,:]
cospec_1s[:,:,:]=cospec[0:int(ntime/2),0:int(nlon/2),:];
cospec_1s[1:int(ntime/2)-1,1:int(nlon/2)-1,:]=2*cospec_1s[1:int(ntime/2)-1,1:int(nlon/2)-1,:]
quadr_1s[:,:,:]=quadr[0:int(ntime/2),0:int(nlon/2),:];
quadr_1s[1:int(ntime/2)-1,1:int(nlon/2)-1,:]=2*quadr_1s[1:int(ntime/2)-1,1:int(nlon/2)-1,:]

peast=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)
pwest=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)
tr=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)
treast=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)
trwest=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)
totF=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)
staF=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)
staP=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)
totP=np.empty((int(ntime/2),int(nlon/2),sizeheat),dtype=complex)

peast[:,:,:]=(powercos_1s[:,:,:]+powersin_1s[:,:,:])/4+quadr_1s[:,:,:]/2
pwest[:,:,:]=(powercos_1s[:,:,:]+powersin_1s[:,:,:])/4-quadr_1s[:,:,:]/2
tr[:,:,:]=np.abs(quadr_1s[:,:,:])
treast[:,:,:]=quadr_1s[:,:,:]
trwest[:,:,:]=-quadr_1s[:,:,:]
totF[:,:,:]=(powercos_1s[:,:,:]+powersin_1s[:,:,:])/2
staF[:,:,:]=(totF[:,:,:]-tr[:,:,:])
staP[:,:,:]=np.sqrt(cospec_1s[:,:,:]*cospec_1s[:,:,:]+((powercos_1s[:,:,:]-powersin_1s[:,:,:])*(powercos_1s[:,:,:]-powersin_1s[:,:,:]))/4)
totP[:,:,:]=staP[:,:,:]+tr[:,:,:]

avepeast=np.mean(peast,axis=2)
avepwest=np.mean(pwest,axis=2)
avetr=np.mean(tr,axis=2)
avetreast=np.mean(treast,axis=2)
avetrwest=np.mean(trwest,axis=2)
avetotF=np.mean(totF,axis=2)
avestaF=np.mean(staF,axis=2)
avestaP=np.mean(staP,axis=2)
avetotP=np.mean(totP,axis=2)

np.savez(outfolder+'/'+outfile,\
         ntime=ntime,nlon=nlon,\
         avepeast=avepeast,avepwest=avepwest,\
         avetr=avetr,avetreast=avetreast,avetrwest=avetrwest,\
         avetotF=avetotF,avestaF=avestaF,\
         avestaP=avestaP,avetotP=avetotP)


