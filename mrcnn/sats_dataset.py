from mrcnn import utils

class SatsDataset(utils.Dataset):
    def scale_coord(self, geom, image):

        scale_x = abs(geom[0] - image.bounds[0]) * abs(image.width / (image.bounds[0] - image.bounds[2]))
        scale_y = abs(geom[1] - image.bounds[3]) * abs(image.height / (image.bounds[1] - image.bounds[3]))


        return scale_x, scale_y
    def preprocessing_image_ms(self, x, mean, std):
        # loop over image bands
        for idx, mean_value in enumerate(mean):
            x[..., idx] -= mean_value
            x[..., idx] /= std[idx]
        return x

    def load_sats(self, image_path, geojson_path):
        from glob import glob
        
        self.add_class("sat", 1, "building")
        
        image_glob = glob(image_path + '*.tif')
        for idx, path in enumerate(image_glob):
            self.add_image("sat", image_id=idx, path=path,
                          jsonPath=geojson_path)
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        from skimage.io import imread
        import numpy as np
        
        # Load image
        input_path = self.image_info[image_id]['path']
        image = np.array(imread(input_path), dtype=float)
        
        bands = [4,2,1]
        
        image = image[:,:,bands]
        
        mean_std_data = np.loadtxt('image_mean_std.txt', delimiter=',')
        mean_std_data = mean_std_data[bands,:]
        image = self.preprocessing_image_ms(image, mean_std_data[:,0], mean_std_data[:,1])
        
        return image        
    
    def load_orig_image(self, image_id):
        """Load the specified image (without stand.) and return a [H,W,3] Numpy array.
        """
        from skimage.io import imread
        import numpy as np
        # Load image
        input_path = self.image_info[image_id]['path']
        image = np.array(imread(input_path), dtype=float)
        
        bands = [4,2,1]
        
        image = image[:,:,bands]
        
        image = (image * 255) / (image.max() + 1e-07)
        
        return image        
    def load_mask(self, image_id):
        import cv2
        import os
        import json
        import rasterio as rio
        import numpy as np
        import scipy.ndimage as ndi
        
        geojson_path = self.image_info[image_id]['jsonPath']
        input_path = self.image_info[image_id]['path']
        
        image_filename = os.path.split(input_path)[-1]
        json_filename = 'buildings' + image_filename[14:-4] + '.geojson'
        geojson_file = os.path.join(geojson_path, json_filename)
    
        #Load JSON
        with open(geojson_file, 'r') as f:
            geo_json = json.load(f)
    
        #Open image to get scale
        image = rio.open(input_path)
        image_shape = image.shape
        #Load and scale all the polygons (buildings)
        polys = []

        for feature in geo_json['features']:
            scaled_coordSet = []
            if feature['geometry']['type'] == 'Polygon':
                for coordinatesSet in feature['geometry']['coordinates']:
                    for coordinates in coordinatesSet:
                        scale_x, scale_y = self.scale_coord(coordinates, image)
                        scaled_coordSet += [[scale_x, scale_y]]

        
            if feature['geometry']['type'] == 'MultiPolygon':
                for polygon in feature['geometry']['coordinates']:
                    for coordinatesSet in polygon:
                        scaled_coord = []
                        for coordinates in coordinatesSet:
                            scale_x, scale_y = self.scale_coord(coordinates, image)
                            scaled_coord += [[scale_x, scale_y]]
                    scaled_coord = np.array(scaled_coord)
                scaled_coordSet += [scaled_coord]

            geom_fixed = np.array(scaled_coordSet, dtype=np.int32)
    
            if geom_fixed.shape[0] != 0:
                polys += [geom_fixed]
        
        polys = np.array(polys)

        mask = np.zeros(image_shape)
        cv2.fillPoly(mask, polys, 1)
    
        mask = mask.reshape(mask.shape[0], mask.shape[1])
        
        segs, count = ndi.label(mask)
        if count == 0:
            maskArr = np.empty([0, 0, 0])
            class_ids = np.empty([0], np.int32)
        else:
            maskArr = np.empty((segs.shape[0], segs.shape[1]))
            class_id_list = []
            for i in range(1, count+1):
                intArr = (segs == i)
                intArr.astype(int)
                maskArr = np.dstack((maskArr, intArr))
                class_id_list += [1]
            maskArr = np.delete(maskArr, 0, axis=2)
            
            class_ids = np.array(class_id_list)
        return maskArr, class_ids