import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import galpy
import astropy.units as u
from ipywidgets import interact, interactive, fixed, interact_manual, HBox, Layout, VBox
import ipywidgets as widgets
from matplotlib.patches import Ellipse

########################################################################
#Initial condition selector widget function - not currently working!
#Ends up with multiple dials

#def initsel(x1=-100, y1=0, vx1=0, vy1=30, phi1=0, m1=1, 
#            x2=100, y2=0, vx2=0, vy2=-30, phi2=0, m2=1):

#    plt.figure(figsize=(6,6))
#    a=plt.subplot(111,aspect='equal')
#    #plt.scatter(x1,y1,c='k',s=100*np.sqrt(m1))
#    e1 = Ellipse((x1,y1), 30*np.sqrt(m1), 30*np.sqrt(m1)*np.cos(phi1*np.pi/180.), 0.)
#    e1.set_clip_box(a.bbox)
#    e1.set_color('k')
#    a.add_artist(e1)
#    plt.quiver(x1,y1,vx1,vy1,color='k',lw=2,angles='xy', scale_units='xy', scale=.5)
#    #plt.scatter(x2,y2,c='r',s=100*np.sqrt(m2))
#    e2 = Ellipse((x2,y2), 30*np.sqrt(m2), 30*np.sqrt(m2)*np.cos(phi2*np.pi/180.), 0.)
#    e2.set_clip_box(a.bbox)
#    e2.set_color('r')
#    a.add_artist(e2)
#    plt.quiver(x2,y2,vx2,vy2,color='r',lw=2,angles='xy', scale_units='xy', scale=.5)
#    plt.xlim(-200,200)
#    plt.ylim(-200,200)
#    plt.xlabel('x [kpc]')
#    plt.ylabel('y [kpc]')

########################################################################
#Setup simulation from widget w
def simsetup(w):
    gal1 = Galaxy(mfac=w['m1'],pos=[w['x1'],w['y1'],0],vel=[w['vx1'],w['vy1'],0],
        rdisk=2.5,phi=w['phi1'],theta=w['theta1'],friction=False,dt=0.1,G=1.)
    gal2 = Galaxy(mfac=w['m2'],pos=[w['x2'],w['y2'],0],vel=[w['vx2'],w['vy2'],0],
        rdisk=2.5,phi=w['phi2'],theta=w['theta2'],friction=False,dt=0.1,G=1.)
    s1 = Stars(gal1, n=1000)
    s2 = Stars(gal2, n=1000)

    return gal1,gal2,s1,s2


########################################################################
#Galaxy class
#halos are log halos - make this adjustable?
class Galaxy():
    def __init__(self,mfac=1,pos=[0,0,0],vel=[0,0,0],rdisk=2.5,phi=0,theta=0,friction=False,dt=0.1,G=1.):
        #initilize
        self.mfac=mfac
        self.pos=np.array(pos)
        self.vel=np.array(vel)
        self.rdisk=rdisk
        self.phi=phi
        self.theta=theta
        #self.halo=halo
        self.friction=friction
        self.dt=dt
        self.G=G
        
        #scale
        self.m = 4.8*mfac
        self.vhalo = 1.*mfac**(.25)
        self.ahalo = 1.*mfac**(.5)
        self.diskSize = self.rdisk*mfac**(.5)
        a2 = -self.m/self.vhalo**2
        a1 = 2*self.ahalo*a2
        a0 = self.ahalo**2*a2
        q = a1/3.0 - a2*a2/9.
        r = (a1*a2-3.*a0)/6. - a2*a2*a2/27.
        s1 = (r + np.sqrt(q**3+r**2))**(1/3.)
        s2 = (r - np.sqrt(q**3+r**2))**(1/3.)
        self.rthalo = (s1+s2)-a2/3
    
    def move(self):
        #update positions and velocities
        self.pos = self.pos + self.vel * self.dt + 0.5 * self.acc * self.dt**2;
        self.vel = self.vel + self.acc * self.dt
         
        return 
    
    def accel(self,pos):
        #compute acceleration from this galaxy at another position
        dpos = pos-self.pos
        try: r = np.sqrt(np.sum(dpos**2,axis=1))
        except:  r = np.sqrt(np.sum(dpos**2))
        accmag=-self.G*self.massEnc(r)/r**2
        try: return (accmag/r)*dpos
        except: return (accmag/r).repeat(3).reshape(len(r),3)*dpos

    def pot(self,pos):
        #compute acceleration from this galaxy at another position
        dpos = self.pos-pos
        r = np.sqrt(np.sum(dpos**2,axis=1))
        return self.G*self.massEnc(r)/r

    def massEnc(self,r):
        #mass enclosed
        r = np.asarray(r)
        scalar_input = False
        if r.ndim == 0:
            r = r[None]  # Makes x 1D
            scalar_input = True
            
        out = np.array([self.m if i > self.rthalo else (self.vhalo**2*i**3/((self.ahalo+i)**2)) for i in r])
        
        if scalar_input:    
            return np.squeeze(out)
        return out
    
    def dens(self,r):
        #relative density for dynamical friction calculation
        r = np.asarray(r)
        scalar_input = False
        if r.ndim == 0:
            r = r[None]  # Makes x 1D
            scalar_input = True
            
        rinner=r*0.99
        router=r*1.01
        minner=self.massEnc(r*0.99)
        mouter=self.massEnc(r*1.01)
        dm=(mouter-minner)
        vol=(4./3.)*np.pi*(router**3-rinner**3)
        density=dm/vol

        if scalar_input:    
            return np.squeeze(density)
        return density
    
    def dynFric(self,pmass,pos,vel):
        #sort of mimics a Chandrasekhar-type expression, returns acceleration
        f=0.5
        lnGamma=3.
        #magnitudes of dv and dx
        v = np.sqrt(np.sum((self.vel-vel)**2,axis=1))
        r = np.sqrt(np.sum((self.pos-pos)**2,axis=1))
        galrho=self.dens(r)
        fricmag=4.0*np.pi*self.G*lnGamma*pmass*galrho*v/((1+v)**3)
        friction = -frictmag*(vel-self.vel)/v
        return friction
        
    def vcirc(self,r):
        #compute circular velocity; calculatess the relative distance and the cicular velocity there
        r = np.asarray(r)
        scalar_input = False
        if r.ndim == 0:
            r = r[None]  # Makes x 1D
            scalar_input = True
        
        vcirc = np.sqrt(self.massEnc(r)/r)
        
        if scalar_input:    
            return np.squeeze(vcirc)
        return vcirc
        
########################################################################
#Star class
#Generated from the initial conditions given to the provided Galaxy class
#currently 1d, using just the area differential to give a density profile; 
#should be done better

class Stars():
    def __init__(self,gal,n=1000):
        cosphi=np.cos(gal.phi*np.pi/180.)
        sinphi=np.sin(gal.phi*np.pi/180.)
        costheta=np.cos(gal.theta*np.pi/180.)
        sintheta=np.sin(gal.theta*np.pi/180.)
        rs = np.random.uniform(low=0., high=gal.rdisk, size=n)
        phis = np.random.uniform(low=0., high=np.pi*2, size=n)
        self.dt = gal.dt
        
        xs = rs*np.cos(phis)
        ys = rs*np.sin(phis)
        zs = np.zeros_like(rs)
        xf= xs*cosphi + ys*sinphi*costheta + zs*sinphi*sintheta
        yf=-xs*sinphi + ys*cosphi*costheta + zs*cosphi*sintheta
        zf=-ys*sintheta + zs*costheta
        self.pos = np.vstack((xf,yf,zf)).T + gal.pos
        
        vcirc=np.sqrt(gal.massEnc(rs)/rs)
        vxs=-vcirc*ys/rs
        vys= vcirc*xs/rs
        vzs= 0.0

        vxf= vxs*cosphi + vys*sinphi*costheta + vzs*sinphi*sintheta
        vyf=-vxs*sinphi + vys*cosphi*costheta + vzs*cosphi*sintheta
        vzf=-vys*sintheta + vzs*costheta
        
        self.vel = np.vstack((vxf,vyf,vzf)).T + gal.vel

    def move(self):
        #update positions and velocities
        self.pos = self.pos + self.vel * self.dt + 0.5 * self.acc * self.dt**2;
        self.vel = self.vel + self.acc * self.dt
         
        return 