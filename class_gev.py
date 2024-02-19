###############
## Libraries ##
###############

import sys,os
import datetime as dt
import itertools as itt

import numpy  as np
import pandas as pd
import xarray as xr

import SDFC as sd
import scipy.stats as sc
import scipy.interpolate as sci
import statsmodels.api as sm
from multiprocessing import Pool, freeze_support

from statsmodels.gam.api import GLMGam, BSplines


#############
## Classes ##
#############


class GEVFixedBoundLinearLink(sd.link.MultivariateLink):##{{{

    def __init__( self , bound ):##{{{
        sd.link.MultivariateLink.__init__( self , n_features = 4 , n_samples = 0 )
        self.bound = bound
    ##}}}

    def transform( self , coef , X ):##{{{
        XX = X[0] if type(X) == list else X
        loc   = coef[0] + coef[1] * XX[:,0]
        scale = np.log( 1 + np.exp( coef[2] + coef[3] * XX[:,0]))
        shape = scale / ( loc - self.bound )
        return loc,scale,shape
    ##}}}

    def jacobian( self , coef , X ):##{{{
        XX   = X[0] if type(X) == list else X
        jac = np.zeros( (3 ,  self.n_samples , self.n_features ) )

        loc   = coef[0] + coef[1] * XX[:,0]
        scale = np.log( 1 + np.exp( coef[2] + coef[3] * XX[:,0]))
        shape = scale / ( loc - self.bound )

        jac[0,:,0] = 1
        jac[1,:,0] = 0
        jac[2,:,0] = - scale / (loc - self.bound)**2

        jac[0,:,1] = XX[:,0]
        jac[1,:,1] = 0
        jac[2,:,1] = - scale * XX[:,0] / (loc - self.bound)**2

        jac[0,:,2] = 0
        jac[1,:,2] = ( np.exp(scale) - 1 ) / np.exp(scale)
        jac[2,:,2] = ( np.exp(scale) - 1 ) / np.exp(scale) * 1 / (loc - self.bound)

        jac[0,:,3] = 0
        jac[1,:,3] = XX[:,0] * ( np.exp(scale) - 1 ) / np.exp(scale)
        jac[2,:,3] = XX[:,0] * ( np.exp(scale) - 1 ) / np.exp(scale) * 1 / (loc - self.bound)

        return jac
    ##}}}

    def valid_point( self , law ):##{{{

        ## Fit by assuming linear case without link functions
        linear_law = type(law)("lmoments")
        l_c = [ c for c in law._rhs.c_global if c is not None ]
        l_c = np.hstack(l_c)
        linear_law.fit( law._Y , c_loc = l_c , c_scale = l_c )
        return linear_law.coef_[:4]
    ##}}}

##}}}

class GEVFixedBoundLinearLink_alt(sd.link.MultivariateLink):##{{{

    def __init__( self , bound ):##{{{
        sd.link.MultivariateLink.__init__( self , n_features = 3 , n_samples = 0 )
        self.bound = bound
    ##}}}

    def transform( self , coef , X ):##{{{
        XX = X[0] if type(X) == list else X
        loc   = coef[0] + coef[1] * XX[:,0]
        scale = np.exp( coef[2] ) + np.zeros(self._n_samples)
        shape = scale / ( loc - self.bound )
        return loc,scale,shape
    ##}}}

    def jacobian( self , coef , X ):##{{{
        XX   = X[0] if type(X) == list else X
        jac = np.zeros( (3 ,  self.n_samples , self.n_features ) )

        loc   = coef[0] + coef[1] * XX[:,0]
        scale = np.exp( coef[2] )
        shape = scale / ( loc - self.bound )

        jac[0,:,0] = 1
        jac[1,:,0] = 0
        jac[2,:,0] = - scale / (loc - self.bound)**2

        jac[0,:,1] = XX[:,0]
        jac[1,:,1] = 0
        jac[2,:,1] = - scale * XX[:,0] / (loc - self.bound)**2

        jac[0,:,2] = 0
        jac[1,:,2] = scale
        jac[2,:,2] = scale / (loc - self.bound)

        return jac
    ##}}}

    def valid_point( self , law ):##{{{

        ## Fit by assuming linear case without link functions
        linear_law = type(law)("lmoments")
        l_c = [ c for c in law._rhs.c_global if c is not None ]
        l_c = np.hstack(l_c)
        linear_law.fit( law._Y , c_loc = l_c )
        return linear_law.coef_[:3]
    ##}}}

##}}}



class GEVFixedBoundStationaryLink(sd.link.MultivariateLink):##{{{

    def __init__( self , bound ):##{{{
        sd.link.MultivariateLink.__init__( self , n_features = 2 , n_samples = 0 )
        self.bound = bound
    ##}}}

    def transform( self , coef , X ):##{{{
        loc   = coef[0] + np.zeros(self._n_samples)
        scale = np.exp(coef[1]) + np.zeros(self._n_samples)
        shape = scale / ( loc - self.bound ) + np.zeros(self._n_samples)
        return loc,scale,shape
    ##}}}

    def jacobian( self , coef , X ):##{{{

        jac = np.zeros( (3 ,  self.n_samples , self.n_features ) )

        loc   = coef[0]
        scale = np.exp(coef[1])
        shape = scale / ( loc - self.bound )

        jac[0,:,0] = 1
        jac[1,:,0] = 0
        jac[2,:,0] = - scale / (loc - self.bound)**2

        jac[0,:,1] = 0
        jac[1,:,1] = scale
        jac[2,:,1] = scale / (loc - self.bound)

        return jac
    ##}}}

    def valid_point( self , law ):##{{{

        ## Fit by assuming linear case without link functions
        linear_law = type(law)("lmoments")
        linear_law.fit( law._Y )
        return linear_law.coef_[:2]
    ##}}}

##}}}

class GEVNonStationaryLogExp(sd.link.MultivariateLink):##{{{

    def __init__( self ):##{{{
        sd.link.MultivariateLink.__init__( self , n_features = 5 , n_samples = 0 )
    ##}}}

    def transform( self , coef , X ):##{{{
        XX = X[0] if type(X) == list else X
        loc   = coef[0] + coef[1] * XX[:,0]
        scale = np.log( 1 + np.exp( coef[2] + coef[3] * XX[:,0]) )
        shape = coef[4] + np.zeros(self._n_samples)
        return loc,scale,shape
    ##}}}

    def jacobian( self , coef , X ):##{{{
        XX   = X[0] if type(X) == list else X
        jac = np.zeros( (3 ,  self.n_samples , self.n_features ) )

        loc   = coef[0] + coef[1] * XX[:,0]
        scale = np.log( 1 + np.exp( coef[2] + coef[3] * XX[:,0]) )
        shape = coef[4]

        jac[0,:,0] = 1
        jac[1,:,0] = 0
        jac[2,:,0] = 0

        jac[0,:,1] = XX[:,0]
        jac[1,:,1] = 0
        jac[2,:,1] = 0

        jac[0,:,2] = 0
        jac[1,:,2] = ( np.exp(scale) - 1 ) / np.exp(scale)
        jac[2,:,2] = 0

        jac[0,:,3] = 0
        jac[1,:,3] = XX[:,0] * ( np.exp(scale) - 1 ) / np.exp(scale)
        jac[2,:,3] = 0
        
        jac[0,:,4] = 0
        jac[1,:,4] = 0
        jac[2,:,4] = 1

        return jac
    ##}}}

    def valid_point( self , law ):##{{{

        ## Fit by assuming linear case without link functions
        linear_law = type(law)("lmoments")
        l_c = [ c for c in law._rhs.c_global if c is not None ]
        l_c = np.hstack(l_c)
        linear_law.fit( law._Y , c_loc = l_c , c_scale = l_c )
        return linear_law.coef_
    ##}}}

##}}}

class GEVNonStationaryLogExp_loc(sd.link.MultivariateLink):##{{{

    def __init__( self , mu0 , mu1 ):##{{{
        sd.link.MultivariateLink.__init__( self , n_features = 3 , n_samples = 0 )
        self.mu0 = mu0
        self.mu1 = mu1
    ##}}}

    def transform( self , coef , X ):##{{{
        XX = X[0] if type(X) == list else X
        loc   = self.mu0 + self.mu1 * XX[:,0]
        scale = np.log( 1 + np.exp( coef[0] + coef[1] * XX[:,0]) )
        shape = coef[2] + np.zeros(self._n_samples)
        return loc,scale,shape
    ##}}}

    def jacobian( self , coef , X ):##{{{
        XX   = X[0] if type(X) == list else X
        jac = np.zeros( (3 ,  self.n_samples , self.n_features ) )

        loc   = self.mu0 + self.mu1 * XX[:,0]
        scale = np.log( 1 + np.exp( coef[0] + coef[1] * XX[:,0]) )
        shape = coef[2]

        jac[0,:,0] = 0
        jac[1,:,0] = ( np.exp(scale) - 1 ) / np.exp(scale)
        jac[2,:,0] = 0

        jac[0,:,1] = 0
        jac[1,:,1] = XX[:,0] * ( np.exp(scale) - 1 ) / np.exp(scale)
        jac[2,:,1] = 0
        
        jac[0,:,2] = 0
        jac[1,:,2] = 0
        jac[2,:,2] = 1

        return jac
    ##}}}

    def valid_point( self , law ):##{{{

        ## Fit by assuming linear case without link functions
        linear_law = type(law)("lmoments")
        l_c = [ c for c in law._rhs.c_global if c is not None ]
        l_c = np.hstack(l_c)
        linear_law.fit( law._Y , c_loc = l_c , c_scale = l_c )
        return linear_law.coef_[2:]
    ##}}}

##}}}

class GEVNonStationaryLogExp_scale(sd.link.MultivariateLink):##{{{

    def __init__( self , sigma0 , sigma1 ):##{{{
        sd.link.MultivariateLink.__init__( self , n_features = 3 , n_samples = 0 )
        self.sigma0 = sigma0
        self.sigma1 = sigma1
    ##}}}

    def transform( self , coef , X ):##{{{
        XX = X[0] if type(X) == list else X
        loc   = coef[0] + coef[1] * XX[:,0]
        scale = np.log( 1 + np.exp( self.sigma0 + self.sigma1 * XX[:,0]) )
        shape = coef[2] + np.zeros(self._n_samples)
        return loc,scale,shape
    ##}}}

    def jacobian( self , coef , X ):##{{{
        XX   = X[0] if type(X) == list else X
        jac = np.zeros( (3 ,  self.n_samples , self.n_features ) )

        loc   = coef[0] + coef[1] * XX[:,0]
        scale = np.log( 1 + np.exp( self.sigma0 + self.sigma1 * XX[:,0]) )
        shape = coef[2]

        jac[0,:,0] = 1
        jac[1,:,0] = 0
        jac[2,:,0] = 0

        jac[0,:,1] = XX[:,0]
        jac[1,:,1] = 0
        jac[2,:,1] = 0
        
        jac[0,:,2] = 0
        jac[1,:,2] = 0
        jac[2,:,2] = 1

        return jac
    ##}}}

    def valid_point( self , law ):##{{{

        ## Fit by assuming linear case without link functions
        linear_law = type(law)("lmoments")
        l_c = [ c for c in law._rhs.c_global if c is not None ]
        l_c = np.hstack(l_c)
        linear_law.fit( law._Y , c_loc = l_c , c_scale = l_c )
        return linear_law.coef_[[0,1,4]]
    ##}}}

##}}}

class GEVNonStationaryLogExp_shape(sd.link.MultivariateLink):##{{{

    def __init__( self , xi ):##{{{
        sd.link.MultivariateLink.__init__( self , n_features = 4 , n_samples = 0 )
        self.xi = xi
    ##}}}

    def transform( self , coef , X ):##{{{
        XX = X[0] if type(X) == list else X
        loc   = coef[0] + coef[1] * XX[:,0]
        scale = np.log( 1 + np.exp( coef[2] + coef[3] * XX[:,0]) )
        shape = self.xi + np.zeros(self._n_samples)
        return loc,scale,shape
    ##}}}

    def jacobian( self , coef , X ):##{{{
        XX   = X[0] if type(X) == list else X
        jac = np.zeros( (3 ,  self.n_samples , self.n_features ) )

        loc   = coef[0] + coef[1] * XX[:,0]
        scale = np.log( 1 + np.exp( coef[2] + coef[3] * XX[:,0]) )
        shape = self.xi

        jac[0,:,0] = 1
        jac[1,:,0] = 0
        jac[2,:,0] = 0

        jac[0,:,1] = XX[:,0]
        jac[1,:,1] = 0
        jac[2,:,1] = 0

        jac[0,:,2] = 0
        jac[1,:,2] = ( np.exp(scale) - 1 ) / np.exp(scale)
        jac[2,:,2] = 0

        jac[0,:,3] = 0
        jac[1,:,3] = XX[:,0] * ( np.exp(scale) - 1 ) / np.exp(scale)
        jac[2,:,3] = 0

        return jac
    ##}}}

    def valid_point( self , law ):##{{{

        ## Fit by assuming linear case without link functions
        linear_law = type(law)("lmoments")
        l_c = [ c for c in law._rhs.c_global if c is not None ]
        l_c = np.hstack(l_c)
        linear_law.fit( law._Y , c_loc = l_c , c_scale = l_c )
        return linear_law.coef_[:4]
    ##}}}

##}}}

