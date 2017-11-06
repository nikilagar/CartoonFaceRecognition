import scipy.io as scio
matInformation = scio.loadmat("IIIT-CFW1.0/IIITCFWdata.mat")
cells = matInformation['IIITCFWdata']
for i in cells['imgName'][0][0][0]:

