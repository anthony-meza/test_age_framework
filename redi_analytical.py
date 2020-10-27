def redi_analytical(ds,kappa,chanloc,v,vpac,La,Lp,modelh,ntime,dt):
    from scipy import signal
    import numpy as np
    #ntime=1080000
    ny=144
    acheck=np.ones(ntime)
    C=np.ones((ny,40000))*3
    Cpac=np.ones((ny,40000))*3
    Cnew=np.ones(ny)*3
    Cold=np.ones(ny)*3
    Cpacnew=np.ones(ny)*6
    Cpacold=np.ones(ny)*6
    Cs=np.zeros(ny)
    Cs[119:]=np.ones(25)


    watl=-np.diff(v)/ds.dyG[10,10].values
    watl[chanloc-1]=0
    wpac=-np.diff(vpac)/ds.dyG[10,10].values

    pacconst=np.zeros((ny))
    pacconst[chanloc]=1

    t=np.arange(0,ntime)

    kappav=2*10**-5

    dy=108086
    timestep=dt*24*3600*365
    for time in range(1, ntime):
        Cnew[1:-1]=(Cold[1:-1]+
                timestep*((kappa[2:]*La[2:]*Cold[2:]*(modelh[1:-1]+modelh[2:])/modelh[2:]/2
                           -kappa[2:]*La[2:]*Cold[1:-1]*(modelh[1:-1]+modelh[2:])/modelh[1:-1]/2- 
                           kappa[1:-1]*La[1:-1]*Cold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                           +kappa[1:-1]*La[1:-1]*Cold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy**2
                          -(v[2:]*(Cold[2:]+Cold[1:-1])/modelh[2:]-v[1:-1]*(Cold[1:-1]+Cold[0:-2])/modelh[1:-1])/dy/2
                         -pacconst[2:]*vpac[chanloc]*(Cold[chanloc-1]+Cold[chanloc])/dy/2/modelh[chanloc]
                          +pacconst[2:]*(kappa[2:]*La[2:]*Cpacold[2:]-kappa[2:]*La[2:]*Cold[1:-1])/dy**2
                           -watl[1:]*Cold[1:-1]/modelh[1:-1]+kappav*La[1:-1]*(Cs[1:-1]-Cold[1:-1]/modelh[1:-1])/modelh[1:-1]))#
    
        Cpacnew[1:-1]=(Cpacold[1:-1]+
                   timestep*((kappa[2:]*Lp[2:]*Cpacold[2:]*(modelh[1:-1]+modelh[2:])/modelh[2:]/2
                              -kappa[2:]*Lp[2:]*Cpacold[1:-1]*(modelh[1:-1]+modelh[2:])/modelh[1:-1]/2-
                              kappa[1:-1]*Lp[1:-1]*Cpacold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                              +kappa[1:-1]*Lp[1:-1]*Cpacold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy**2
                            -(vpac[2:]*(Cpacold[2:]+Cpacold[1:-1])/modelh[2:]-vpac[1:-1]*
                              (Cpacold[1:-1]+Cpacold[0:-2])/modelh[1:-1])/dy/2
                            -wpac[1:]*Cpacold[1:-1]/modelh[1:-1]+kappav*Lp[1:-1]*(-Cpacold[1:-1]/modelh[1:-1])/modelh[1:-1]))#
        Cnew[0]=0
        Cnew[-1]=modelh[-1]
        Cpacnew[-1]=Cpacnew[-2]
        Cold=Cnew
        Cpacold=Cpacnew
        Cpacold[0:chanloc]=Cold[0:chanloc]
        timeend=time-ntime+40000
        if timeend>=0:
            C[:,timeend]=Cnew
            Cpac[:,timeend]=Cpacnew
    diffusion_term_atl=-((kappa[1:-1]*La[1:-1]*Cold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                           -kappa[1:-1]*La[1:-1]*Cold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy)
    meridional_adv_atl=-(-(v[1:-1]*(Cold[1:-1]+Cold[0:-2])/modelh[1:-1])/2)
    diffusion_term_pac=-((kappa[1:-1]*Lp[1:-1]*Cpacold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                              -kappa[1:-1]*Lp[1:-1]*Cpacold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy)
    meridional_adv_pac=-(-vpac[1:-1]*(Cpacold[1:-1]+Cpacold[0:-2])/modelh[1:-1]/2)
    vertical_adv=-np.cumsum(watl[1:]*Cold[1:-1]/modelh[1:-1]*dy)
    vertical_diff=-np.cumsum(kappav*La[1:-1]*(Cs[1:-1]-Cold[1:-1]/modelh[1:-1])/modelh[1:-1]*dy)
    return C,Cpac,diffusion_term_atl,meridional_adv_atl,diffusion_term_pac,meridional_adv_pac


def redi_analytical_AABW(ds,kappa,chanloc,v,vpac,La,Lp,modelh,ntime,dt):
    from scipy import signal
    import numpy as np
    #ntime=1080000
    ny=144
    acheck=np.ones(ntime)
    C=np.ones((ny,40000))*3
    Cpac=np.ones((ny,40000))*3
    Cnew=np.ones(ny)*3
    Cold=np.ones(ny)*3
    Cpacnew=np.ones(ny)*3
    Cpacold=np.ones(ny)*3
    Cs=np.zeros(ny)
    Cs[0:8]=np.ones(8)

    watl=-np.diff(v)/ds.dyG[10,10].values
    watl[chanloc-1]=0
    wpac=-np.diff(vpac)/ds.dyG[10,10].values

    pacconst=np.zeros((ny))
    pacconst[chanloc]=1

    t=np.arange(0,ntime)

    kappav=2*10**-5

    dy=108086
    timestep=dt*24*3600*365
    for time in range(1, ntime):
        Cnew[1:-1]=(Cold[1:-1]+
                timestep*((kappa[2:]*La[2:]*Cold[2:]*(modelh[1:-1]+modelh[2:])/modelh[2:]/2
                           -kappa[2:]*La[2:]*Cold[1:-1]*(modelh[1:-1]+modelh[2:])/modelh[1:-1]/2- 
                           kappa[1:-1]*La[1:-1]*Cold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                           +kappa[1:-1]*La[1:-1]*Cold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy**2
                          -(v[2:]*(Cold[2:]+Cold[1:-1])/modelh[2:]-v[1:-1]*(Cold[1:-1]+Cold[0:-2])/modelh[1:-1])/dy/2
                         -pacconst[2:]*vpac[chanloc]*(Cold[chanloc-1]+Cold[chanloc])/dy/2/modelh[chanloc]
                          +pacconst[2:]*(kappa[2:]*La[2:]*Cpacold[2:]-kappa[2:]*La[2:]*Cold[1:-1])/dy**2
                           -watl[1:]*Cold[1:-1]/modelh[1:-1]+kappav*La[1:-1]*(Cs[1:-1]-Cold[1:-1]/modelh[1:-1])/modelh[1:-1]))#
    
        Cpacnew[1:-1]=(Cpacold[1:-1]+
                   timestep*((kappa[2:]*Lp[2:]*Cpacold[2:]*(modelh[1:-1]+modelh[2:])/modelh[2:]/2
                              -kappa[2:]*Lp[2:]*Cpacold[1:-1]*(modelh[1:-1]+modelh[2:])/modelh[1:-1]/2-
                              kappa[1:-1]*Lp[1:-1]*Cpacold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                              +kappa[1:-1]*Lp[1:-1]*Cpacold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy**2
                            -(vpac[2:]*(Cpacold[2:]+Cpacold[1:-1])/modelh[2:]-vpac[1:-1]*
                              (Cpacold[1:-1]+Cpacold[0:-2])/modelh[1:-1])/dy/2
                            -wpac[1:]*Cpacold[1:-1]/modelh[1:-1]+kappav*Lp[1:-1]*(Cs[1:-1]-Cpacold[1:-1]/modelh[1:-1])/modelh[1:-1]))#
        Cnew[0]=modelh[0]
        Cnew[-1]=0
        Cpacnew[-1]=Cpacnew[-2]
        Cold=Cnew
        Cpacold=Cpacnew
        Cpacold[0:chanloc]=Cold[0:chanloc]
        timeend=time-ntime+40000

        if timeend>=0:
            C[:,timeend]=Cnew
            Cpac[:,timeend]=Cpacnew
    return C,Cpac


def redi_analytical_atlLGM(ds,kappa0,chanloc,v,modelh,modelh1,modelh2,ntime):
    from scipy import signal
    import numpy as np
    #ntime=1080000
    ny=144
    acheck=np.ones(ntime)
    C=np.ones((ny,40000))*3
    Cnew=np.ones(ny)*3
    Cold=np.ones(ny)*3
    aatl=Cold[1:-1]
    Cs=np.zeros(ny)
    Cs[119:]=np.ones(25)
    watl=-np.diff(v)/ds.dyG[10,10].values
    watl[chanloc-1]=watl[chanloc+1]
    atlconst=np.ones((ny))*1/3
    atlconst[0:chanloc]=1
    atlconst[chanloc+1:]=1
    t=np.arange(0,ntime)
    kappa=kappa0*np.ones(ny)
    kappav=2*10**-5

    dy=108086
    timestep=0.01*24*3600*365
    for time in range(1, ntime):
        Cnew[1:-1]=(Cold[1:-1]+
                timestep*((kappa[2:]*atlconst[2:]*Cold[2:]*(modelh[1:-1]+modelh[2:])/modelh[2:]/2
                           -kappa[2:]*atlconst[2:]*Cold[1:-1]*(modelh[1:-1]+modelh[2:])/modelh[1:-1]/2- 
                           kappa[1:-1]*Cold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                           +kappa[1:-1]*Cold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy**2
                          -(atlconst[2:]*v[2:]*(Cold[2:]+Cold[1:-1])/modelh[2:]-v[1:-1]*(Cold[1:-1]+Cold[0:-2])/modelh[1:-1])/dy/2
                           -watl[1:]*Cold[1:-1]/modelh[1:-1]+kappav*aatl*np.exp(watl[72]*(-4000+modelh[1:-1])/kappav)))#
        Cnew[0]=0
        Cnew[-1]=modelh[-1]
        Cold=Cnew
        timeend=time-ntime+40000
        aatl=(Cold[1:-1]-Cs[1:-1]*modelh[1:-1])*(watl[72]/kappav)/(-modelh[1:-1]+kappav/watl[72]*(
            np.exp(watl[72]*(-modelh1[1:-1])/kappav)-np.exp(watl[72]*(-modelh2[1:-1])/kappav)))
        acheck[time]=aatl[72]
        if timeend>=0:
            C[:,timeend]=Cnew
    return C

def redi_analytical_LGM(ds,kappa,chanloc,v,vpac,vatlll,vpacll,modelh,ll,ntime,dt):
    from scipy import signal
    import numpy as np
    #ntime=1080000
    ny=144
    acheck=np.ones(ntime)
    C=np.ones((ny))*3
    Cpac=np.ones((ny))*3
    Cnew=np.ones(ny)*3
    Cold=np.ones(ny)*3
    Cpacnew=np.ones(ny)*3
    Cpacold=np.ones(ny)*3
    Cll=np.ones((ny))*3
    Cpacll=np.ones((ny))*3
    Cnewll=np.ones(ny)*3
    Coldll=np.ones(ny)*3
    Cpacnewll=np.ones(ny)*3
    Cpacoldll=np.ones(ny)*3
    Cs=np.zeros(ny)
    Cs[119:]=np.ones(25)
    aatl=Cold[1:-1]
    apac=Cpacold[1:-1]
    
    wll=diff_kappa_v(-ll)
    wll[wll>0]=0
    
    modelhll=4000-ll
    watl=-np.diff(v)/ds.dyG[10,10].values
    watl[chanloc-1]=0
    watlll=np.diff(vatlll)/ds.dyG[10,10].values
    watlll[chanloc-1]=0
    
    wpac=-np.diff(vpac)/ds.dyG[10,10].values
    wpacll=-np.diff(vpacll)/ds.dyG[10,10].values
    
    atlconst=np.ones((ny))*1/3
    atlconst[0:chanloc]=1
    atlconst[chanloc+1:]=1
    pacconst=np.zeros((ny))
    pacconst[chanloc]=2/3
    pacconst2=np.ones((ny))*2/3
    pacconst2[0:chanloc]=1
    pacconst2[chanloc+1:]=1
    t=np.arange(0,ntime)
    #modelh=(4000-1000*(np.tanh((np.arange(0,144)-14)/5)+1)/2-100)
    #modelh=(4000-800*(np.tanh((np.arange(0,144)-14)/5)+1)/2-100)
    #kappa=kappa0*np.ones(ny)
    kappav=2*10**-5

    dy=108086
    timestep=dt*24*3600*365
    for time in range(1, ntime):
        Cnew[1:-1]=(Cold[1:-1]+
                timestep*((kappa[2:]*atlconst[2:]*Cold[2:]*(modelh[1:-1]+modelh[2:])/modelh[2:]/2
                           -kappa[2:]*atlconst[2:]*Cold[1:-1]*(modelh[1:-1]+modelh[2:])/modelh[1:-1]/2- 
                           kappa[1:-1]*Cold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                           +kappa[1:-1]*Cold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy**2
                          -(atlconst[2:]*v[2:]*(Cold[2:]+Cold[1:-1])/modelh[2:]
                            -v[1:-1]*(Cold[1:-1]+Cold[0:-2])/modelh[1:-1])/dy/2
                         -pacconst[2:]*vpac[chanloc]*(Cpacold[chanloc+1]+Cpacold[chanloc])/dy/2/modelh[chanloc]
                          +pacconst[2:]*(kappa[2:]*Cpacold[2:]-kappa[2:]*Cold[1:-1])/dy**2
                           -(watl[1:]-watlll[1:])*Cold[1:-1]/modelh[1:-1]-watlll[1:]*Cold[1:-1]/modelh[1:-1]
                          +wll[1:-1]*(Cold[1:-1]/modelh[1:-1]-Coldll[1:-1]/modelhll[1:-1])))#
        
        Cnewll[1:-1]=(Coldll[1:-1]+
                timestep*((kappa[2:]*atlconst[2:]*Coldll[2:]*(modelhll[1:-1]+modelhll[2:])/modelhll[2:]/2
                           -kappa[2:]*atlconst[2:]*Coldll[1:-1]*(modelhll[1:-1]+modelhll[2:])/modelhll[1:-1]/2- 
                           kappa[1:-1]*Coldll[1:-1]*(modelhll[1:-1]+modelhll[0:-2])/modelhll[1:-1]/2
                           +kappa[1:-1]*Coldll[0:-2]*(modelhll[1:-1]+modelhll[0:-2])/modelhll[0:-2]/2)/dy**2
                          -(atlconst[2:]*vatlll[2:]*(Coldll[2:]+Coldll[1:-1])/modelhll[2:]-vatlll[1:-1]*
                          (Coldll[1:-1]+Coldll[0:-2])/modelhll[1:-1])/dy/2
                         -pacconst[2:]*vpacll[chanloc]*(Coldll[chanloc-1]+Coldll[chanloc])/dy/2/modelhll[chanloc]
                          +pacconst[2:]*(kappa[2:]*Cpacoldll[2:]-kappa[2:]*Coldll[1:-1])/dy**2
                          +watlll[1:]*Cold[1:-1]/modelh[1:-1]
                           -wll[1:-1]*(Cold[1:-1]/modelh[1:-1]-Coldll[1:-1]/modelhll[1:-1])))
    
        Cpacnew[1:-1]=(Cpacold[1:-1]+
                   timestep*((kappa[2:]*pacconst2[2:]*Cpacold[2:]*(modelh[1:-1]+modelh[2:])/modelh[2:]/2
                              -kappa[2:]*pacconst2[2:]*Cpacold[1:-1]*(modelh[1:-1]+modelh[2:])/modelh[1:-1]/2-
                              kappa[1:-1]*Cpacold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                              +kappa[1:-1]*Cpacold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy**2
                            -(vpac[2:]*(Cpacold[2:]+Cpacold[1:-1])/modelh[2:]-vpac[1:-1]*
                              (Cpacold[1:-1]+Cpacold[0:-2])/modelh[1:-1])/dy/2
                            -(wpac[1:]+wpacll[1:])*Cpacold[1:-1]/modelh[1:-1]
                             +wpacll[1:]*Cpacoldll[1:-1]/modelhll[1:-1]
                             +wll[1:-1]*(Cpacold[1:-1]/modelh[1:-1]-Cpacoldll[1:-1]/modelhll[1:-1])))#
        
        Cpacnewll[1:-1]=(Cpacoldll[1:-1]+
                   timestep*((kappa[2:]*pacconst2[2:]*Cpacoldll[2:]*(modelhll[1:-1]+modelhll[2:])/modelhll[2:]/2
                              -kappa[2:]*pacconst2[2:]*Cpacoldll[1:-1]*(modelhll[1:-1]+modelhll[2:])/modelhll[1:-1]/2-
                              kappa[1:-1]*Cpacoldll[1:-1]*(modelhll[1:-1]+modelhll[0:-2])/modelhll[1:-1]/2
                              +kappa[1:-1]*Cpacoldll[0:-2]*(modelhll[1:-1]+modelhll[0:-2])/modelhll[0:-2]/2)/dy**2
                            -(vpacll[2:]*(Cpacoldll[2:]+Cpacoldll[1:-1])/modelhll[2:]-vpacll[1:-1]*
                              (Cpacoldll[1:-1]+Cpacoldll[0:-2])/modelhll[1:-1])/dy/2
                            +(wpac[1:]+wpacll[1:])*Cpacold[1:-1]/modelh[1:-1]
                             -wpacll[1:]*Cpacoldll[1:-1]/modelhll[1:-1])
                             -wll[1:-1]*(Cpacold[1:-1]/modelh[1:-1]-Cpacoldll[1:-1]/modelhll[1:-1]))#
        Cnew[0]=0
        Cnewll[0]=0
        Cnew[-1]=modelh[-1]
        Cnewll[-1]=Cnewll[-2]
        Cpacnew[-1]=Cpacnew[-2]
        Cpacnewll[-1]=Cpacnewll[-2]
        Cold=Cnew
        Coldll=Cnewll
        Cpacold=Cpacnew
        Cpacoldll=Cpacnewll
        Cpacold[0:chanloc]=Cold[0:chanloc]
        Cpacoldll[0:chanloc]=Coldll[0:chanloc]
        if time==ntime-1:
            C=Cnew
            Cpac=Cpacnew
            Cll=Cnewll
            Cpacll=Cpacnewll
    diffusion_term_atl=-((kappa[1:-1]*Cold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                           -kappa[1:-1]*Cold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy)
    meridional_adv_atl=-(-(v[1:-1]*(Cold[1:-1]+Cold[0:-2])/modelh[1:-1])/2)
    diffusion_term_pac=-((kappa[1:-1]*Cpacold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                              -kappa[1:-1]*Cpacold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy)
    meridional_adv_pac=-(-vpac[1:-1]*(Cpacold[1:-1]+Cpacold[0:-2])/modelh[1:-1]/2)
    vertical_adv=-np.cumsum(watl[1:]*Cold[1:-1]/modelh[1:-1]*dy)
    vertical_diff=-np.cumsum(kappav*aatl*np.exp(watl[72]*(-4000+modelh[1:-1])/kappav)*dy)
    return C,Cpac,Cll,Cpacll,diffusion_term_atl,meridional_adv_atl,diffusion_term_pac,meridional_adv_pac,wll


def redi_analytical_LGM2(ds,kappa,chanloc,v,vpac,vatlll,vpacll,La,Lp,modelh,ll_var,kappaz,ntime,dt):
    from scipy import signal
    import numpy as np
    #ntime=1080000
    ny=144
    acheck=np.ones(ntime)
    C=np.ones((ny))*3
    Cpac=np.ones((ny))*3
    Cnew=np.ones(ny)*3
    Cold=np.ones(ny)*3
    Cpacnew=np.ones(ny)*300
    Cpacold=np.ones(ny)*300
    Cll=np.ones((ny))*300
    Cpacll=np.ones((ny))*300
    Cnewll=np.ones(ny)*300
    Coldll=np.ones(ny)*300
    Coldll2=np.ones(ny)*300
    Cpacnewll=np.ones(ny)*300
    Cpacoldll=np.ones(ny)*300
    Cpacoldll2=np.ones(ny)*300
    Cs=np.zeros(ny)
    Cs[119:]=np.ones(25)#*0.8
    
    #wll=diff_kappa_v(-ll)
    #wll[wll>0]=0
    
    modelhll=4000-ll_var
    
    
    
    watlll=np.diff(vatlll)/ds.dyG[10,10].values
    watlll[chanloc-1]=watlll[chanloc-2]
    watlll_down=np.zeros((ny-1))
    watlll_down[watlll>0]=watlll[watlll>0]
    watlll_up=watlll-watlll_down
    #watlll2[72:]=0
    
    watl=-np.diff(v)/ds.dyG[10,10].values-watlll
    watl[chanloc-1]=watl[chanloc-2]
    watl_down=np.zeros((ny-1))
    watl_down[watl>0]=watl[watl>0]
    watl_up=watl-watl_down
    
    wpacll=-np.diff(vpacll)/ds.dyG[10,10].values
    wpacll_down=np.zeros((ny-1))
    wpacll_down[wpacll<0]=wpacll[wpacll<0]
    wpacll_up=wpacll-wpacll_down
    
    wpac=-np.diff(vpac)/ds.dyG[10,10].values+wpacll
    wpac_down=np.zeros((ny-1))
    wpac_down[wpac<0]=wpac[wpac<0]
    wpac_up=wpac-wpac_down
    
    atlconst=np.ones((ny))
    atlconst[0:chanloc]=1
    atlconst[chanloc+1:]=1
    pacconst=np.zeros((ny))
    pacconst[chanloc]=1

    t=np.arange(0,ntime)
    
    kappav=2*10**-5
    #kappaz=2*10**(-5)*np.ones(ny)#+10**-2*(1+np.tanh((-4000+ll+30)/30))/2+(2*10**-4+2*10**-3*(10**((4000-ll-4000)/2000)))*(1-np.tanh((-4000+ll+2000)/200))/2
    
    
    maxll=122
    
    
    modelhll2=modelhll.copy()#(modelh+modelhll)/2
    modelhll2[maxll-1:]=99999999
    
    modelhll3=modelhll.copy()#(modelh+modelhll)/2
    modelhll3[modelhll<70]=99999999
    modelhll3[maxll-1:]=99999999
    
    modelh2=modelh.copy()
    #modelh2[modelhll<200]=200
    modelh2[maxll-1:]=99999999
    
    dy=108086#ds.dyG[10,10].values
    timestep=dt*24*3600*365
    for time in range(1, ntime):
        atlllterm=-watlll_down[1:]*Cold[1:-1]/modelh[1:-1]-watlll_up[1:]*Coldll2[1:-1]/modelhll2[1:-1]
        atlllterm[maxll:]=0
        Cnew[1:-1]=(Cold[1:-1]+
                timestep*((kappa[2:]*La[2:]*atlconst[2:]*Cold[2:]*(modelh[1:-1]+modelh[2:])/modelh[2:]/2
                           -kappa[2:]*La[2:]*atlconst[2:]*Cold[1:-1]*(modelh[1:-1]+modelh[2:])/modelh[1:-1]/2- 
                           kappa[1:-1]*La[1:-1]*Cold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                           +kappa[1:-1]*La[1:-1]*Cold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy**2
                          -(atlconst[2:]*v[2:]*(Cold[2:]+Cold[1:-1])/(modelh[2:]+modelh[1:-1])
                            -v[1:-1]*(Cold[1:-1]+Cold[0:-2])/(modelh[1:-1]+modelh[0:-2]))/dy
                         -pacconst[2:]*vpac[chanloc]*(Cpacold[chanloc+1]+Cpacold[chanloc])/dy/2/modelh[chanloc]
                          +pacconst[2:]*(kappa[2:]*Lp[2:]*Cpacold[2:]-kappa[2:]*Lp[2:]*Cold[1:-1])/dy**2
                           -watl_down[1:]*Cold[1:-1]/modelh[1:-1]
                          +atlllterm
                          +kappav*La[1:-1]*(Cs[1:-1]-Cold[1:-1]/modelh[1:-1])/modelh[1:-1]
                          -kappaz[1:-1]*La[1:-1]*(Cold[1:-1]/modelh2[1:-1]-Coldll2[1:-1]/modelhll2[1:-1])/modelhll3[1:-1]))
        
        Cnewll[1:maxll]=(Coldll[1:maxll]+
                timestep*((kappa[2:maxll+1]*La[2:maxll+1]*atlconst[2:maxll+1]*Coldll[2:maxll+1]*
                           (modelhll[1:maxll]+modelhll[2:maxll+1])/modelhll[2:maxll+1]/2
                           -kappa[2:maxll+1]*La[2:maxll+1]*atlconst[2:maxll+1]*Coldll[1:maxll]*
                           (modelhll[1:maxll]+modelhll[2:maxll+1])/modelhll[1:maxll]/2- 
                           kappa[1:maxll]*La[1:maxll]*Coldll[1:maxll]*(modelhll[1:maxll]+modelhll[0:maxll-1])/modelhll[1:maxll]/2
                           +kappa[1:maxll]*La[1:maxll]*Coldll[0:maxll-1]*(modelhll[1:maxll]+modelhll[0:maxll-1])/modelhll[0:maxll-1]/2)/dy**2
                          -(atlconst[2:maxll+1]*vatlll[2:maxll+1]*(Coldll[2:maxll+1]+Coldll[1:maxll]
                                                                  )/(modelhll[2:maxll+1]+modelhll[1:maxll])-vatlll[1:maxll]*
                          (Coldll[1:maxll]+Coldll[0:maxll-1])/(modelhll[1:maxll]+modelhll[0:maxll-1]))/dy
                         -pacconst[2:maxll+1]*vpacll[chanloc]*(Coldll[chanloc-1]+Coldll[chanloc])/dy/2/modelhll[chanloc]
                          +pacconst[2:maxll+1]*(kappa[2:maxll+1]*Lp[2:maxll+1]*Cpacoldll[2:maxll+1]-kappa[2:maxll+1]*Lp[2:maxll+1]*Coldll[1:maxll]
                                               )/dy**2
                        +watlll_down[1:maxll]*Cold[1:maxll]/modelh[1:maxll]+watlll_up[1:maxll]*Coldll[1:maxll]/modelhll[1:maxll]
                         +kappaz[1:maxll]*La[1:maxll]*(Cold[1:maxll]/modelh[1:maxll]-Coldll[1:maxll]/modelhll[1:maxll])/modelhll3[1:maxll]))
        
    
        pacllterm=wpacll_up[1:]*Cpacoldll2[1:-1]/modelhll2[1:-1]
        pacllterm[maxll:]=0
        Cpacnew[1:-1]=(Cpacold[1:-1]+
                   timestep*((kappa[2:]*Lp[2:]*Cpacold[2:]*(modelh[1:-1]+modelh[2:])/modelh[2:]/2
                              -kappa[2:]*Lp[2:]*Cpacold[1:-1]*(modelh[1:-1]+modelh[2:])/modelh[1:-1]/2-
                              kappa[1:-1]*Lp[1:-1]*Cpacold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2
                              +kappa[1:-1]*Lp[1:-1]*Cpacold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy**2
                            -(vpac[2:]*(Cpacold[2:]+Cpacold[1:-1])/(modelh[2:]+modelh[1:-1])-vpac[1:-1]*
                              (Cpacold[1:-1]+Cpacold[0:-2])/(modelh[1:-1]+modelh[0:-2]))/dy
                            -wpac_up[1:]*Cpacold[1:-1]/modelh[1:-1]
                             +pacllterm
                             +wpacll_down[1:]*Cpacold[1:-1]/modelh[1:-1]
                             +kappav*Lp[1:-1]*(-Cpacold[1:-1]/modelh[1:-1])/modelh[1:-1]
                             -kappaz[1:-1]*Lp[1:-1]*(Cpacold[1:-1]/modelh2[1:-1]-Cpacoldll2[1:-1]
                                               /modelhll2[1:-1])/modelhll3[1:-1]))
        
        Cpacnewll[1:maxll]=(Cpacoldll[1:maxll]+
                   timestep*((kappa[2:maxll+1]*Lp[2:maxll+1]*Cpacoldll[2:maxll+1]
                              *(modelhll[1:maxll]+modelhll[2:maxll+1])/modelhll[2:maxll+1]/2
                              -kappa[2:maxll+1]*Lp[2:maxll+1]*Cpacoldll[1:maxll]*
                              (modelhll[1:maxll]+modelhll[2:maxll+1])/modelhll[1:maxll]/2-
                              kappa[1:maxll]*Lp[1:maxll]*Cpacoldll[1:maxll]*(modelhll[1:maxll]+modelhll[0:maxll-1])/modelhll[1:maxll]/2
                              +kappa[1:maxll]*Lp[1:maxll]*Cpacoldll[0:maxll-1]*
                              (modelhll[1:maxll]+modelhll[0:maxll-1])/modelhll[0:maxll-1]/2)/dy**2
                            -(vpacll[2:maxll+1]*(Cpacoldll[2:maxll+1]+Cpacoldll[1:maxll]
                                                                )/(modelhll[2:maxll+1]+modelhll[1:maxll])-vpacll[1:maxll]*
                              (Cpacoldll[1:maxll]+Cpacoldll[0:maxll-1])/(modelhll[1:maxll]+modelhll[0:maxll-1]))/dy
                          #  +wpac_down[1:]*Cpacold[1:-1]/modelh[1:-1]
                             -wpacll_down[1:maxll]*Cpacold[1:maxll]/modelh[1:maxll]
                             -wpacll_up[1:maxll]*Cpacoldll[1:maxll]/modelhll[1:maxll]
                             +kappaz[1:maxll]*Lp[1:maxll]*(Cpacold[1:maxll]/modelh[1:maxll]-Cpacoldll[1:maxll]
                                               /modelhll[1:maxll])/modelhll3[1:maxll]))
        Cnew[0]=0
        Cnewll[0]=0
        Cnew[-1]=modelh[-1]
        Cnewll[maxll:]=Cnewll[maxll-1]*modelhll[maxll:]/modelhll[maxll-1]
        Cpacnew[-1]=Cpacnew[-2]*modelh[-1]/modelh[-2]#Cpacnewll[-1]*modelh[-1]/modelhll[-1]
        Cpacnewll[maxll:]=Cpacnewll[maxll-1]*modelhll[maxll:]/modelhll[maxll-1]
        Cold=Cnew
        Coldll=Cnewll
        Coldll2=Cnewll.copy()
        Coldll2[maxll:]=0
        Cpacold=Cpacnew
        Cpacoldll=Cpacnewll
        Cpacoldll2=Cpacnewll.copy()
        Cpacoldll2[maxll:]=0
        Cpacold[0:chanloc]=Cold[0:chanloc]
        Cpacoldll[0:chanloc]=Coldll[0:chanloc]
        
        if time==ntime-1:
            C=Cnew
            Cpac=Cpacnew
            Cll=Cnewll
            Cpacll=Cpacnewll
    diffusion_term_atl=(kappa[1:-1]*Cold[1:-1]*(modelh[1:-1]+modelh[0:-2])/modelh[1:-1]/2- 
                           kappa[1:-1]*Cold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy
    meridional_adv_atl= -(-(v[1:-1]*(Cold[1:-1]+Cold[0:-2])/modelh[1:-1])/2)
    diffusion_term_pac=(kappa[1:-1]*Cpacold[2:]*(modelh[1:-1]+modelh[0:-2])/modelh[2:]/2-
                              +kappa[1:-1]*Cpacold[0:-2]*(modelh[1:-1]+modelh[0:-2])/modelh[0:-2]/2)/dy
    meridional_adv_pac=-(-vpac[1:-1]*(Cpacold[1:-1]+Cpacold[0:-2])/modelh[1:-1]/2)
    diffusion_term_atlll=(kappa[1:-1]*Coldll[1:-1]*(modelhll[1:-1]+modelhll[0:-2])/modelhll[1:-1]/2- 
                           kappa[1:-1]*Coldll[0:-2]*(modelhll[1:-1]+modelhll[0:-2])/modelhll[0:-2]/2)/dy
    meridional_adv_atlll= -(-(vatlll[1:-1]*(Coldll[1:-1]+Coldll[0:-2])/modelhll[1:-1])/2)
    diffusion_term_pacll=(kappa[1:-1]*Cpacoldll[2:]*(modelhll[1:-1]+modelhll[0:-2])/modelhll[2:]/2-
                              +kappa[1:-1]*Cpacoldll[0:-2]*(modelhll[1:-1]+modelhll[0:-2])/modelhll[0:-2]/2)/dy
    meridional_adv_pacll=-(-vpacll[1:-1]*(Cpacoldll[1:-1]+Cpacoldll[0:-2])/modelhll[1:-1]/2)
    return C,Cpac,Cll,Cpacll,diffusion_term_atl,meridional_adv_atl,diffusion_term_pac,meridional_adv_pac,diffusion_term_atlll,meridional_adv_atlll,diffusion_term_pacll,meridional_adv_pacll

def diff_kappa_v(z):
    import numpy as np
    dkappadz=(10**-2/(np.cosh((z+30)/30)**2)/60-
          (2*10**-3*np.log([10])*(10**((-z-4000)/2000)))*(1-np.tanh((z+2000)/200))/2/2000-
        (2*10**-4+2*10**-3*(10**((-z-4000)/2000)))/np.cosh((z+2000)/200)**2/400)
    return dkappadz