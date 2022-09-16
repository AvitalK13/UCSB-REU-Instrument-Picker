#!/usr/bin/env python
# coding: utf-8

# In[12]:


#To do list:
#Add more instruments!
#Make more modular
#Add BEX models
#Add JWST stuff

#get_speckle takes lists for both arguments, get_post takes single separation... check and revise in code that I'm using them correctly


# In[364]:


#Model dictionary
import numpy as np
model_choice = {'COND': 'model.AMES-Cond-2000.M-0.0.MKO.Vega', 
                    'Dusty': 'model.AMES-dusty.M-0.0.MKO.Vega',
                    'BT': 'model.BT-Settl.M-0.0.MKO.Vega',
               'BEX': 'BEX_evol_mags_-2_MH_0.00.dat'}
    
instrument_filter_table = {'GPI':['H'],
                           'CHARIS':['H'],
                           'NIRC2':['Lp'],
                            'JWST':['F560W', 'F770W', 'F1000W', 'F1130W',
                                    'F1280W', 'F1500W', 'F1800W', 'F2100W',
                                    'F2550W', 'FND', 'F070W', 'F090W', 'F115W',
                                    'F140M', 'F150W', 'F162M', 'F164N', 'F150W2',
                                    'F182M', 'F187N', 'F200W', 'F210M', 'F212N',
                                    'F250M', 'F277W', 'F300M', 'F322W2', 'F323N',
                                    'F335M', 'F360M', 'F356W', 'F405N', 'F410M',
                                    'F430M', 'F444W', 'F460M', 'F466N', 'F470N',
                                    'F480M', 'F158M', 'F380M', 'F140X', 'F110W',
                                    'F070LP', 'F100LP', 'F170LP', 'F290LP', 'CLEAR', 'Opaque']}

#Instruments
instrument_choice = {'CHARIS': np.loadtxt('SCExAO Redo.csv', delimiter=',', dtype=str),
                         'GPI': np.loadtxt('GPI 9 Mag.csv',delimiter=',',dtype=str)}


# In[352]:


instrument_list = ['CHARIS','GPI']


# In[428]:


def get_speckle_noise(separations, ao_mag):
        import numpy as np
        import scipy.interpolate as si
        '''
        Source 2: Vanessa Bailey https://arxiv.org/pdf/1609.08689.pdf
        
        The code currently uses contrast data from Vanessa Bailey's paper. The contrast at separations larger than 0.8" is assumed to be the same as at 0.8".  The contrast at separations smaller than 0.25" is extrapolated.
        
        Inputs: 
        separations     - A list of separations at which to calculate the speckle noise [float list length n] in units of arcsecond
        ao_mag          - The magnitude in the ao band, here assumed to be I-band

        Outputs: 
        get_speckle_noise - Either an array of length [n,1] if only one wavelength passed, or shape [n,m]

        '''
        #Read in the separation v contrast file        
        contrasts_table = np.loadtxt('gpi_contrasts.csv', dtype = float, skiprows = 2) # Source 2
        separation_data = contrasts_table[0:3,0]
        contrast_data = contrasts_table[:,1]
        get_speckle_noise = []

       #Make an interpolation function to find the contrast at other points
        for i in separations:
            if i > 0.8:
                i = 0.8 #We assume that at separations greater than 0.8", the contrast is the same as at 0.8"
            if ao_mag < 2.0:
                contrasts = contrast_data[0:3]
            elif ao_mag >= 2.0 and ao_mag <= 3.0:
                contrasts = contrast_data[0:3]
            elif ao_mag > 3.0 and ao_mag <= 4.0:
                contrasts = contrast_data[3:6]
            elif ao_mag > 4.0 and ao_mag <= 5.0:
                contrasts = contrast_data[6:9]
            elif ao_mag > 5.0 and ao_mag <= 6.0:
                contrasts = contrast_data[9:12]
            elif ao_mag > 6.0 and ao_mag <= 7.0:
                contrasts = contrast_data[12:15]
            elif ao_mag > 7.0 and ao_mag <= 8.0:
                contrasts = contrast_data[15:18]
            elif ao_mag > 8.0 and ao_mag <= 9.0:
                contrasts = contrast_data[18:21]
            elif ao_mag > 9.0 and ao_mag <= 10.0:
                contrasts = contrast_data[21:24]
            elif ao_mag > 10.0:
                contrasts = contrast_data[21:24]

            f = si.interp1d(separation_data, contrasts, fill_value = "extrapolate") #We extrapolate contrasts at separations smaller than 0.25"
            interpolated_contrast = f(i)/5 # Dividing by 5 because we used 5-sigma noise values and we want 1-sigma
            get_speckle_noise = np.append(get_speckle_noise, interpolated_contrast)
            
        return get_speckle_noise


# In[429]:


def get_post_processing_gain(AngSep): #Changed to return post_processing_gain_mean instead of 
    import numpy as np
    import scipy.interpolate as si
    '''
    Function to return the post processing gain based on the stellar I-band magnitude
    
    Source 4: Vanessa Bailey https://arxiv.org/pdf/1609.08689.pdf; Interpolation based on initial contrast to final contrast ratios from Figure 4  
    
    Takes angular separation in arcseconds as input
    '''
    interp_post_processing = si.interp1d([.25,.4,.8],[15.48,13.3,14.12],bounds_error = False, fill_value = ([15.48],[13.3]))   #Changed to [15.48,13.3,14.12]
    interp_post_processing_std = si.interp1d([.25,.4,.8],[.54,.48,.46],bounds_error = False, fill_value = ([.54],[.46]))
    post_processing_gain_mean = interp_post_processing(AngSep)   #Deleted '.value'
    post_processing_gain_std = interp_post_processing_std(AngSep)   #Deleted '.value'
    post_processing_gain = np.random.lognormal(post_processing_gain_mean,post_processing_gain_std)
    return post_processing_gain_mean


# In[423]:


def find_masses(instrument,model,stellar_apparent_magnitude,age,distance):

    '''
    Inputs:

        Instrument: The instrument with which we are testing for planets
        Model: The evolutionary model used to interpolate the input data
        Stellar_apparent_magnitude: The apparent magnitude of the target star
        Age: The age in Gyr of the target star
        Distance: The distance in parsecs to the target star
    '''
    #Import packages
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import contrast_utils as conut
    import pandas as pd

    #Import models
    cond_model = np.genfromtxt('model.AMES-Cond-2000.M-0.0.MKO.Vega_astrotab.txt')
    dusty_model = np.genfromtxt('model.AMES-dusty.M-0.0.MKO.Vega_astrotab.txt')
    bt_model = np.genfromtxt('model.BT-Settl.M-0.0.MKO.Vega_astrotab.txt')
    bex_model = np.genfromtxt('BEX_evol_mags_-2_MH_0.00.dat')
    
    #COND model data
    cond_model_age = cond_model[1:,12]
    cond_model_mass = cond_model[1:,0] #in Jupiter masses
    cond_model_app_mag = cond_model[1:,10]
    cond_bands = {'J':'col8','H':'col9','Ks':'col10','Lp':'col11','Mp':'col12'}
    
    #Dusty model data
    dusty_model_age = dusty_model[1:,12]
    dusty_model_mass = dusty_model[1:,0] #in Jupiter masses
    dusty_model_app_mag = dusty_model[1:,10] #Isn't this the Lp band?
    dusty_bands = {'J':'col8','H':'col9','Ks':'col10','Lp':'col11','Mp':'col12'}
    
    #BT model data
    bt_model_age = bt_model[1:,12]
    bt_model_mass = bt_model[1:,0] #in Jupiter masses
    bt_model_app_mag = bt_model[1:,10]
    bt_bands = {'J':'col8','H':'col9','Ks':'col10','Lp':'col11','Mp':'col12'}
    
    #BEX model data
    bex_model_age = (10**bex_model[0:,0])/(10**9)
    bex_model_mass = ((bex_model[0:,1])*0.0031469) #in Jupiter masses
    bex_bands = {'F115W':'18:F115W', 'F150W':'19:F150W', 'F200W':'20:F200W',
                      'F277W':'21:F277W', 'F356W':'22:F356W', 'F444W':'23:F444W',
                      'F560W':'24:F560W', 'F770W':'25:F770W', 'F1000W':'26:F1000W',
                      'F1280W':'27:1280W', 'F1500W':'28:1500W', 'F1800W':'29:1800W',
                      'F2100W':'30:2100W', 'F2550':'31:F2550W', 'Y':'34:SPHEREY',
                      'J':'35:SPHEREJ', 'H':'36:SPHEREH', 'Ks':'37:SPHEREKs', 'J2':'38:SPHEREJ2',
                      'J3':'39:SPHEREJ3', 'H2':'40:SPHEREH2', 'H3':'41:SPHEREH3',
                      'K1':'42:SPHEREK1', 'K2':'43:SPHEREK2'}
    
    if model=='COND':
        bands = cond_bands
    elif model=='Dusty':
        bands = dusty_bands
    elif model=='BT':
        bands = bt_bands
    if model=='BEX':
        bands = []
        for i in bex_bands:
            bands.append(bex_bands[instrument_filter_table[instrument][0]])

        
    #Instruments
    instrument_choice = {'CHARIS': np.loadtxt('SCExAO Redo.csv', delimiter=',', dtype=str),
                         'GPI': np.loadtxt('GPI 9 Mag.csv',delimiter=',',dtype=str)}
    
    if instrument == 'CHARIS':
        scexao_good = np.loadtxt('SCExAO Redo.csv', delimiter=',', dtype=str) #Change to bring it from dictionary?
        scex_contrast = scexao_good[0:,1]
        scex_separation = scexao_good[0:,0]
        contrast = []
        for i in scex_contrast:
            contrast.append(float(i))
        separation = []
        for i in scex_separation:
            separation.append(float(i))
    elif instrument == 'GPI':
        gpi = np.loadtxt('gpi_contrasts.csv',dtype=float,skiprows=2)
        gpi_seps = np.arange(0.15,1.4,0.1)
        gpi_contrasts = get_speckle_noise(gpi_seps,stellar_apparent_magnitude)*5/get_post_processing_gain(gpi_seps)
        contrast = []
        for i in gpi_contrasts:
            contrast.append(float(i))
        separation = []
        for i in gpi_seps:
            separation.append(float(i))
    
    model_data = conut.read_and_format_txtfile(model_choice[str(model)])
    
    #Inputs
    stellar_app_mag = float(stellar_apparent_magnitude)
    distance = float(distance)
    star_age = float(age)
    parallax = distance/1000 #in milliarcseconds
    
    #Calculate planet apparent magnitude
    planet_app_mag = []

    for i in contrast:
        fl=float(i)
        mag = fl*stellar_app_mag
        planet_app_mag.append(mag) #Do we even need this??

    #Call mass from mag function
    mass_function = conut.get_mass_func_from_mag(model_data,parallax,star_age,band=bands[instrument_filter_table[instrument][0]],no_interp=False)

    #Calculate masses
    needed_mags = []
    for i in contrast:
        fl=float(i)
        mags = stellar_app_mag-2.5*np.log10(fl)
        needed_mags.append(mags)                #What are these mags? How are they different than the planet mags?
        
    mass_calc = mass_function(needed_mags)
    
    print('{} magnitudes: {}'.format(instrument,needed_mags))

    return mass_calc


# In[412]:


bex_model = np.genfromtxt('BEX_evol_mags_-2_MH_0.00.dat')
bex_bands = {'F115W':'18:F115W', 'F150W':'19:F150W', 'F200W':'20:F200W',
                      'F277W':'21:F277W', 'F356W':'22:F356W', 'F444W':'23:F444W',
                      'F560W':'24:F560W', 'F770W':'25:F770W', 'F1000W':'26:F1000W',
                      'F1280W':'27:1280W', 'F1500W':'28:1500W', 'F1800W':'29:1800W',
                      'F2100W':'30:2100W', 'F2550':'31:F2550W', 'Y':'34:SPHEREY',
                      'J':'35:SPHEREJ', 'H':'36:SPHEREH', 'Ks':'37:SPHEREKs', 'J2':'38:SPHEREJ2',
                      'J3':'39:SPHEREJ3', 'H2':'40:SPHEREH2', 'H3':'41:SPHEREH3',
                      'K1':'42:SPHEREK1', 'K2':'43:SPHEREK2'}

instrument_filter_table = {'GPI':['H'],
                           'CHARIS':['H'],
                           'NIRC2':['Lp'],
                            'JWST':['F560W', 'F770W', 'F1000W', 'F1130W',
                                    'F1280W', 'F1500W', 'F1800W', 'F2100W',
                                    'F2550W', 'FND', 'F070W', 'F090W', 'F115W',
                                    'F140M', 'F150W', 'F162M', 'F164N', 'F150W2',
                                    'F182M', 'F187N', 'F200W', 'F210M', 'F212N',
                                    'F250M', 'F277W', 'F300M', 'F322W2', 'F323N',
                                    'F335M', 'F360M', 'F356W', 'F405N', 'F410M',
                                    'F430M', 'F444W', 'F460M', 'F466N', 'F470N',
                                    'F480M', 'F158M', 'F380M', 'F140X', 'F110W',
                                    'F070LP', 'F100LP', 'F170LP', 'F290LP', 'CLEAR', 'Opaque']}

instrument = 'GPI'
print(bex_bands[instrument_filter_table[instrument][0]])


# In[413]:


find_masses('GPI','COND',5,0.021,59)
find_masses('CHARIS','BEX',3,0.008,67)


# In[436]:


import matplotlib.pyplot as plt
import numpy as np

contrasts_table = np.loadtxt('gpi_contrasts.csv', dtype = float, skiprows = 2) # Source 2
separation_data = contrasts_table[0:3,0]
contrast_data = contrasts_table[:,1]
cont = np.linspace(max(contrast_data),min(contrast_data),len(separation_data))

scexao_good = np.loadtxt('SCExAO Redo.csv', delimiter=',', dtype=str) #Change to bring it from dictionary?
scex_contrast = scexao_good[0:,1]
scex_separation = scexao_good[0:,0]
contrast = []
for i in scex_contrast:
    contrast.append(float(i))
separation = []
for i in scex_separation:
    separation.append(float(i))

plt.figure()
plt.semilogy(separation_data,cont)
plt.semilogy(separation,contrast)
plt.title('Raw Contrast (GPI and CHARIS)')
plt.xlabel('Separation (arcseconds)')
plt.ylabel('Contrast')
plt.show()


# In[424]:


def compare_instruments(model,stellar_apparent_magnitude,age,distance,instruments):
    #Import packages
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import contrast_utils as conut
    
    for i in instruments:
        if i == 'CHARIS':
            scexao_mass_list = (find_masses(i,model,stellar_apparent_magnitude,age,distance))
        elif i == 'GPI':
            gpi_mass_list = (find_masses(i,model,stellar_apparent_magnitude,age,distance))
   
    #Instruments
    instrument_choice = {'CHARIS': np.loadtxt('SCExAO Redo.csv', delimiter=',', dtype=str),
                         'GPI': np.loadtxt('GPI 9 Mag.csv',delimiter=',',dtype=str)}
    
    #Create plots
    fig,ax=plt.subplots(1,2,figsize=(20,8))
    ax[0].set_title('Contrast vs. Separation for {}'.format(instruments))
    ax[1].set_title('Mass vs. Separation for {}'.format(instruments))
    ax[0].set_xlabel('Separation (arcseconds)')
    ax[0].set_ylabel('Contrast')
    ax[1].set_xlabel('Separation (arcseconds)')
    ax[1].set_ylabel('Mass (Mj)')
    
    if 'CHARIS' in instruments:
        scexao_good = np.loadtxt('SCExAO Redo.csv', delimiter=',', dtype=str) #Change to bring it from dictionary
        scex_contrast = scexao_good[0:,1]
        scex_separation = scexao_good[0:,0]
        contrast = []
        for i in scex_contrast:
            contrast.append(float(i))
        separation = []
        for i in scex_separation:
            separation.append(float(i))
        ax[0].semilogy(separation,contrast,marker='*',color='orange',label='CHARIS',linestyle='None')
        ax[1].plot(separation,scexao_mass_list,marker='*',color='orange',label='CHARIS',linestyle='None')
        
        
    if 'GPI' in instruments:
        gpi = np.loadtxt('gpi_contrasts.csv',dtype=float,skiprows=2)
        gpi_seps = np.arange(0.15,1.4,0.1)
        gpi_contrasts = get_speckle_noise(gpi_seps,stellar_apparent_magnitude)*5/get_post_processing_gain(gpi_seps)
        contrast = []
        for i in gpi_contrasts:
            contrast.append(float(i))
        separation = []
        for i in gpi_seps:
            separation.append(float(i))
        ax[0].semilogy(separation,contrast,marker='v',color='teal',label='GPI',linestyle='None')
        ax[1].plot(separation,gpi_mass_list,marker='v',color='teal',label='GPI',linestyle='None')
    ax[0].legend()
    ax[1].legend()
    
    return


# In[425]:


def instrument_picker(model,stellar_apparent_magnitude,age,distance,instruments):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    compare_instruments(model,stellar_apparent_magnitude,age,distance,instruments)
    
    return


# In[426]:


instrument_picker('Dusty',3,0.008,67,instrument_list)


# In[359]:


instrument_picker('BEX',9,0.06,37,instrument_list)


# In[ ]:


#Not the actual code
#Must hard code bands because the files are so different and I can't get them to easily connect
    #Find which column in .Vega is the band, then assign the column number from astrotab
   #cond_data = np.loadtxt('model.AMES-Cond-2000.M-0.0.MKO.Vega', delimiter=',', dtype=str)
    #cond_headers = cond_data[2]
    #cond_band_h = cond_headers.find('H')
    #print(cond_band_h)
    #print(cond_data[76])

