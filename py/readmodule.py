import numpy as np
import operator

###############################################################################
#                             INSTRUCTIONS
#                            --------------
# 
# To read in data you first have to call
# 
# >> data = read()
# then
# >> data.readfile(filename)
# 
# readfile() returns True/False so you can check the data was actually
# read in or if the query was empty.
# You can also only read between certain columns (useful if you have spliced 
# multiple data sets together) by calling
# 
# >> data.readfile(filename, first=0, last=10) 
# 
# which would read columns 0-9 (I use the python notation so it doesn't
# include the last column). If you want to read the last column just call
# it without the last argument
# 
# >> data.readfile(filename, first=10)
# 
# When you want to retrieve the data you have to use dictionary notation
# 
# >> haloIDs = data['haloId']
# 
# If you want to know what the column names are (say you want to loop over
# them) you call
# 
# >> data.keys()
# 
# 
# I added some more functionality so it can be called a little like sql
# (getindex, where, sort, write):
# 
# find the index in the arrays which have name==value, returns an 
# array
# >> index = data.getindex('name', value)
# 
# similar to sql where, returns a new read class, the comparison is 
# either lt, le, eq, ne, ge, gt, isfinite, isnan, isinf or <,<=,==,!=,>=,> (as a string)
# >> newData = data.where('name', 'lt', value)
# >> newData = data.where('name', '<', value)
# OR
# value is name of data key and compares the value of 'name1' against 'name2' for each object
# >> newData = data.where('name1', '<', 'name2')
# 
# similar to sql sort, rearranges all the arrays in the same way
# I haven't worked out how to do multiple levels of sorting but its 
# probably not necessary
# >> data.sort('name', direction='asc') # direction='desc'
# 
# write to file
# >> data.write(filename)
# 
# Also you can set an empty class then write data to it yourself:
# 
# >> data = read()
# >> data['name'] = [...] # sets it as an np.array in the class so not 
#                         # necessary to do it here 
#
# An 'add' class has been written so that two classes (with the same data types)
# can be merged. Is called as
# >> newclass = class1 + class2
#
#
# Note:
# Any column with 'ID', 'Id', 'np', or 'snapnum' in the column name are saved
# as numpy.int64; all other columns are numpy.float64
#
# In order to append to a class, you must create a new class then add to the two classes.
# Alternatively if speed is an issue, only write to a class once all data is appended together.
#
###############################################################################

class read:
    '''
    A class for reading in data from the Millennium database

    Makes use of dictionaries so that an arbitrary number of columns with 
    arbitrary names can be read in. Each set of data can then be accessed with
    a call <read>['column_name']
    To find what data is available, use the function <read>.keys()
    '''

    op = {}
    op['lt'] = operator.lt
    op['<'] = operator.lt
    op['le'] = operator.le
    op['<='] = operator.le
    op['eq'] = operator.eq
    op['=='] = operator.eq
    op['ne'] = operator.ne
    op['!='] = operator.ne
    op['ge'] = operator.ge
    op['>='] = operator.ge
    op['gt'] = operator.gt
    op['>'] = operator.gt
    op['isnan'] = np.isnan
    op['isfinite'] = np.isfinite
    op['isinf'] = np.isinf

    #
    # private functions
    #

    def __init__(self):
        self._registry={} # names of available data types
        self._data={}     # the data
        self._N=0         # length of data arrays, gets defined in __setitem__

    def __setitem__(self,name,item):
        if self._N==0:
            self._N = len(item)
        elif self._N!=0:
            # All items must be same length!
            if len(item) != self._N:
                raise RuntimeError( 'Length of %s is not the same as items already set!'%(name))
        self._data[name] = np.array(item)
        self._registry[name] = {}

    def __getitem__(self,name):
        if name in self._data:
            return self._data[name]
        else:
            raise KeyError( "Data type "+name+" is not valid, check <read>.keys() for defined data")

    def __add__(self, other):
        # python does funny things when copying classes (can still point to data in another class)
        # so have to be careful here
        if len(other.keys())==0:
            self_copy = read()
            for item in self.keys():
                self_copy[item] = self._data[item]
            return self_copy
        elif len(self.keys())==0:
            other_copy = read()
            for item in other.keys():
                other_copy[item] = other[item]
            return other_copy
        elif self.viewkeys()!=other.viewkeys():
            raise RuntimeError( 'Cannot add classes, keys not equal')
        new = read()
        for item in self.keys():
            new[item] = np.append(self._data[item], other[item])
            #new[item] = np.array(list(self._data[item])+list(other[item]))
        return new


    #
    # public functions
    #

    def keys(self):
        '''Returns a list of available data types'''
        return self._registry.keys()

    def viewkeys(self):
        '''Returns a list of available data types'''
        return self._registry.viewkeys()

    def N(self):
        '''Number of items in arrays'''
        return self._N

    def readfile(self,filename,first=0,last=0,sep=','):
        '''
        Read in the data
        
        first and last specify which columns to take data from (using pythonic list notation)
        If last==-1 all columns after first are used
        sep: delimiter between strings (None for whitespace)
        '''
        column_headings_read = False
        listin = []
        for line in open(filename,'r'):
            li=line.strip()

            if li=='': continue # empty line
        
            if not li.startswith("#"):
                line = line.partition('#')[0] 

                # first line of data after commented section is a list of 
                # column headings. This is used for the dictionary names
                if not column_headings_read:
                    column_headings_read = True
                    if last==0: last=len(line.rstrip().split(sep))
                    column_headings = line.rstrip().split(sep)[first:last]

                    # Clean whitespace from headings
                    for i in range(len(column_headings)):
                        column_headings[i] = column_headings[i].strip()
                    continue
                   
                listin.append([token for token in line.rstrip().split(sep)][first:last])

        if not column_headings_read:
            raise RuntimeError( 'Column headings not read from file')

        if len(listin)==0:
            raise IOError( 'Error: No data in file %s'%(filename))

        listin = np.asarray(listin)  

        for i in range(len(column_headings)):
            if column_headings[i] in ['Image_ID', 'Image_face','Image_edge','Image_box']: 
                continue
            if 'ID' in column_headings[i] or 'Id' in column_headings[i] or column_headings[i]=='np' or 'snapnum' in column_headings[i].lower() or 'GroupNumber' in column_headings[i]:
                # need to define as type int64 otherwise get roundoff error with haloId
                self[column_headings[i]] = np.array(listin[:,i]).astype(np.float64).astype(np.int64)
            elif column_headings[i]=='mergedSpurious':
                self[column_headings[i]] = np.array(listin[:,i]).astype(str)
            else:
                self[column_headings[i]] = np.array(listin[:,i]).astype(np.float64)

    #@TODO: could rewrite this to take more than one comparison. 
    # i.e. find number of comparisons then loop over them, could recursively call where()
    def where(self, name, cmpr, value=[]): 
        '''
        Finds all items where classobjectname[name] cmpr value
        and cmpr is in the set of comparisons lt(<), le(<=), eq(==), ne(!=), 
        ge(>=), gt(>), isnan(np.isnan), isfinite(np.isfinite), isinf(np.isinf), between.

        isnan, isfinite and isinf take no 'value' input (only None).

        between takes two input values (lower and upper) and is of form: value[0] <= object <= value[1].
        value must be len=2 array when cmpr='between'

        Output is a new readmodule.read() class
        '''
        newobject = read()
        if cmpr=='between':
            if len(value)==0:
                raise RuntimeError('value must be len=2 array when cmpr=between')
            mask1 = self._data[name] >= value[0]
            mask2 = self._data[name] <= value[1]
            mask = mask1*mask2
            for item in self.keys():
                newobject[item] = self._data[item][mask]
            return newobject
        if cmpr not in self.op.keys():
            raise RuntimeError( 'Error in read.where(): comparison operator %s not defined' % cmpr)
        if cmpr in ['isnan','isfinite','isinf']:
            value=None
        if value in self.keys():
            mask = self.op[cmpr](self._data[name], self._data[value])
        else: 
            mask = np.where(self.op[cmpr](self._data[name],value))[0] # where returns tuple
        for item in self.keys():
            newobject[item] = self._data[item][mask]
        return newobject

    def getindex(self, name, value):
        '''
        Finds the indices of items where classobjectname[name]==value
        Returns an array
        '''
        try:
            idx = np.where(self._data[name]==value)[0]
        except:
            # TODO: maybe should raise an error here?
            return None
        return idx

    def write(self,filename,header='',colheadings=[]):
        '''
        Write data in class to a file mimicking Millennium database output
        header is written first in file
        colheadings gives an option of list of strings for column order (need not be all headings)
        '''
        f = open(filename, 'w')
        if len(header)>0:
            f.write(header)
        if len(colheadings)>0:
            # check colheadings is ok
            for heading in colheadings:
                if heading not in self.keys():
                    raise RuntimeError( "Column heading '%s' not in key list" %(heading))
        else:
            colheadings = self.keys()
        f.write( ','.join(colheadings)+'\n' ) # headings
        for i in range(self._N):
            line = []
            for name in colheadings:
                line.append(str(self._data[name][i]))
            f.write(','.join(line)+'\n')
        f.write('#OK\n')
        f.close()


    def sort(self,name,direction='asc'):
        '''
        Sort all arrays accoring to classojectname[name] in the given direction
        Possible sorting directions are 'asc'(ascending) and 'desc'(descending)
        '''
        if direction.lower()=='asc': reverse=False
        elif direction.lower()=='desc': reverse=True
        else: 
            raise RuntimeError( "Sort direction '%s' not recognised" % (direction))

        r = zip(self._data[name], np.arange(self._N))
        arr = sorted(arr, key=lambda arr: arr[0], reverse=reverse)
        arr, sort_key = zip(*arr)
        sort_key = np.array(sort_key)
        del arr

        for key in self.keys():
            self._data[key] = self._data[key][sort_key]

    #@TODO would it be better to allow index to be a list?
    def delete(self, index):
        '''
        Delete all data correspoding to the given index
        index is an int
        '''
        for key in self.keys():
            self._data[key] = np.delete(self._data[key],index)

        self._N = len(self._data[self.keys()[0]])

    def top(self, num):
        '''
        Similar to SQL TOP. Returns to top NUM items in the class
        '''
        newobject = read()
        for item in self.keys():
            newobject[item] = self._data[item][:num]
        return newobject

    def pop(self, key):
        '''
        Remove data with given key. Similar to pop() for dicts
        '''
        self._registry.pop(key, None)
        self._data.pop(key, None)


