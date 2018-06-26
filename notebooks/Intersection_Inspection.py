
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('data/Crashes_in_DC_larger.csv', low_memory=False)

df.columns = [x.lower() for x in df.columns]
df.drop(columns=['locerror','todate'], inplace=True)
df.columns

# Fill in some missing x and y values.
df.x = df.longitude
df.y = df.latitude

df.dropna(how='any', axis=0, inplace=True)

# Reset the index in the reduced df so we avoid index out of bounds errors.
df.reset_index(inplace=True, drop=True) # avoid index out of bounds

# Correct an x-variable with the wrong sign (only one instance).
df.loc[df.x > 0]['x'] = df.x * -1

# Convert ward from string to a numeric value
df['ward_number'] = df.ward.str.split(' ', expand=True)[1]
df.ward_number = pd.to_numeric(df.ward_number)


# ### Experimentation
# Pick a point (some row) and start experimenting with street directions and angles.
#
# Goal is to find a way to identify the nearest roadway, find points along that nearest roadway, and calculate the direction of that nearest roadway.  Additionally, calculate the direction of my point's roadway.
#
# Finally, compute the angle between the two roads, and determine if it is acute or normal.  Maybe add two fields: an angle measurement for each row (with respect to its nearest roadway), and an acute/not-acute boolean.

def get_location_product(row):
    ''' return row.x * row.y'''
    return row.x * row.y

#df['loc_prod'] = df.apply(get_location_product, axis=1)

# Sort rows by loc_prod, see how many of them are nearby each other.
#sorted_df = df.sort_values('loc_prod')


# In[13]:


#sorted_df.head(3)


# In[14]:


#sorted_df[:].plot(x='x', y='y', kind='scatter', alpha=.05, figsize=(8,8))

df['my_street_max'] = 0
df['my_street_min'] = 0
df['other_street_max'] = 0
df['other_street_min'] = 0


# Make a helper function (get_street_ends) to identify the rows containing the max x and the min x for any other row.

# In[76]:


def distance(row):
    ''' may need to rework this to use as an apply function. '''
    return math.sqrt((row.x - my_row.x)**2 + (row.y - my_row.y)**2)

#other823df['x_proximity'] = other823df.apply(distance, axis=1)

def get_street_ends(street_points, eval_row):
    '''
    pass list of row indices for nearby locations to the evaluation row. eval_row is a row from df
    under evaluation.
    return max and min rows based on sorting, proximity to row under eval.
    '''
    my_row = eval_row

    sorted_close_df = df.iloc[street_points].copy(deep=True)
    sorted_close_df['x_proximity'] = 0
    sorted_close_df['x_proximity'] = sorted_close_df.apply(distance, axis=1)
    sorted_close_df.sort_values(by='x_proximity')
    sorted_close_df = sorted_close_df.head() # take top 5 closest rows

    # get the roadwaysegid of the most prominent nearby street
    other_roadwaysegid = int(sorted_close_df['roadwaysegid'].value_counts().head(1).index[0])

    # sorting on x-coordinate, get the max row and the min row for the other_roadwaysegid rows
    my_max = sorted_close_df.loc[sorted_close_df.roadwaysegid == other_roadwaysegid].sort_values(by='x').head(1)
    my_min = sorted_close_df.loc[sorted_close_df.roadwaysegid == other_roadwaysegid].sort_values(by='x').tail(1)
    return my_max, my_min

#my_row = df.iloc[0]
#some_max, some_min = get_street_ends([823, 613, 757], df.iloc[823])


# In[79]:


#some_max
#some_min


def divide_streets(rows):
    '''
    row_index_list is a list of the row indices we want to study.  May run very slowly if we try to
    process the whole data set.

    Divide nearby rows based on streetsegid; if it's the one under study, append to my_street_points.
    TODO: Calculate max and min rows for mine and other, add to dataframe.
    '''
    #close_indices_dict = {}  # For each row of study, get list of other row indices nearby
    for index in rows.closest_pts:
        #close_indices_dict[index] = get_closest_points(df.iloc[index])

        my_streetsegid = df.iloc[index]['streetsegid']
        my_street_points = []
        other_street_points = []

        #for row in close_indices_dict[index]: # divide the nearby rows into mine and others
        for row in rows.closest_points:
            if df.iloc[row]['streetsegid'] == my_streetsegid:
                my_street_points.append(row)
            else:
                other_street_points.append(row)

        #print('getting min, max')
        my_max, my_min = get_street_ends(my_street_points, df.iloc[index])
        #print(my_max, my_min)
        o_max, o_min = get_street_ends(other_street_points, df.iloc[index])
        return [my_max, my_min, o_max, o_min]


# In[101]:


#df.closest_pts
#df.iloc[10000]
#df.closest_pts[df.closest_pts.isna()]

#get_ipython().run_cell_magic('time', '', "df.columns\ndf[['my_street_max','my_street_min','other_street_max','other_street_min']] = df.apply(divide_streets, axis=1)")

#df.columns


# ### Driver for generating angles
#
# Determine the angle between the nearest intersecting road.  Add it to the dataframe as a new field.  Try modeling with it, and see if it helps improve accuracy.
#
# Reference: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249

# In[ ]:


df['my_min'] = 0
df['my_max'] = 0
df['other_min'] = 0
df['other_max'] = 0


# In[ ]:


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#divide_streets(df[:30].dropna())


import gmplot

# Fatalities involved?
df['fatal'] = (df.fatal_pedestrian >=1) | (df.fatal_bicyclist >= 1) | (df.fatal_driver >= 1)

def get_coords(frame):
    ''' Return longitude and latitude coordinates from a dataframe'''
    locations = frame[['x','y']].dropna()
    longs = locations.x.tolist()
    lats = locations.y.tolist()
    return longs, lats

#fatal_longs, fatal_lats = get_coords(df[df.fatal == True])
#f_cycle_longs, f_cycle_lats = get_coords(df[df.fatal_bicyclist == True])
#f_ped_longs, f_ped_lats = get_coords(df[df.fatal_pedestrian == True])
#f_driver_longs, f_driver_lats = get_coords(df[df.fatal_driver == True])
#other_longs, other_lats = get_coords(other823df)

#gmap.scatter(df_lats, df_longs, color='red', size=10, marker=False, alpha=0.5) # sample of all accidents (gray)
#gmap.scatter(other_lats, other_longs, color='blue',size=10, marker=False, alpha=0.5) # Any fatality (magenta)

#title = 'X' # Bicycle Fatalities
#gmap.coloricon = 'http://www.googlemapsmarkers.com/v1/' + title + '/%s/FFFFFF/001211/'
#gmap.marker(df.iloc[823].y, df.iloc[823].x, c='m', title=title)
#gmap.scatter(df.iloc[823].y, df.iloc[823].x, color='red', marker=True, size=50, alpha=5)
#gmap.draw('one_segment_sketch.html')


# In[ ]:


#df.isnull().sum()


# In[ ]:


#index823df = df.iloc[[823, 613, 757, 589, 520, 573, 963, 422, 527, 650, 671, 705, 505, 770, 496, 1000, 127196, 468, 775, 755, 2233, 932, 703, 797, 466, 682, 558, 969, 706, 2256, 634, 500, 456, 660, 2245, 856, 1004, 568, 958, 860, 989, 886, 980, 2258, 799, 652, 549]]
#other823df = df.iloc[[24086, 24259, 22712, 24419, 24270, 24413, 24365, 24306, 1731, 1823, 1711, 1818, 1935, 1973, 1987, 24239, 24232]]

# Which streetsegid is the closest to my datapoint?
import math
closest = 0
mypoint = df.iloc[823]
#other823df['x_proximity'] = math.sqrt((other823df.x - mypoint.x)**2 +
#                                      (other823df.y - mypoint.y)**2)

def distance(row):
    return math.sqrt((row.x - mypoint.x)**2 + (row.y - mypoint.y)**2)

df['closest_pts'] = 0
def get_closest_points(row):
    '''
    for each row under study, add the closest points to the row.
    WARNING - runs slow, so use subsets, not a whole data set at a time.
    '''
    closest_indices = []
    grab = 1000
    max_grab = 0
    while len(closest_indices) < 12 and grab < df.shape[0]:
        if grab > max_grab:
            max_grab = grab
        df_closex = df.iloc[(df['x'] - row.x).abs().argsort()[:grab]]
        #df_closex = df_closex[df_closex.streetsegid != streetsegid] # get points along our street as well.
        x_indices = df_closex.index.tolist()

        df_closey = df.iloc[(df['y'] - row.y).abs().argsort()[:grab]]
        #df_closey = df_closey[df_closey.streetsegid != streetsegid]
        y_indices = df_closey.index.tolist()

        closest_indices = [indx for indx in x_indices if indx in y_indices]
        grab += grab
        df['closest_pts'] = closest_indices

    #print(max_grab)
    return closest_indices

#other823df['x_proximity'] = other823df.apply(distance, axis=1)


# for the 'other streets' rows, sort by distance to my row
#sorted_other_close_df = other823df[['address','nearestintstreetname','routeid','roadwaysegid','streetsegid','offintersection','x_proximity']].sort_values(by='x_proximity')
#sorted_other_close_df = sorted_other_close_df.head() # take the top 5 closest rows
#df.iloc[sorted_other_close_df.index[0]].roadwaysegid # the roadwaysegid of the closest cross street (11028)

# get the roadwaysegid of the most prominent nearby street
#other_roadwaysegid = int(sorted_other_close_df['roadwaysegid'].value_counts().head(1).index[0])

# sorting on x-coordinate, get the max row and the min row for the other_roadwaysegid rows
#othermax = other823df.loc[other823df.roadwaysegid == other_roadwaysegid].sort_values(by='x').head(1)
#othermax

#othermin = other823df.loc[other823df.roadwaysegid == other_roadwaysegid].sort_values(by='x').tail(1)
#othermin

# define my other road vector
#x = (othermax.x.values - othermin.x.values)
#y = (othermax.y.values - othermin.y.values)
#other_road_unit_vector = (x, y)/np.linalg.norm((x,y))
#print(x, y, other_road_unit_vector)
