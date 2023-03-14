import glob

datasetFolder = "NTU_RGB+D_Samples/"
scenesFolder = datasetFolder+"RGB/"
depthsFolder = datasetFolder+"Depth Map/"

AVIs = sorted(glob.glob(scenesFolder+"/*.avi"))

for AVI in AVIs:
    filename = AVI.split('/')[2]
    filename = filename.split('_')[0]
    print(AVI)

    depthMaps = sorted(glob.glob(depthsFolder+filename+"/*.png"))
    frameCounter = 0
    for depthMap in depthMaps:
        print(filename+"_"+str(frameCounter+1).zfill(5))
        frameCounter+=1