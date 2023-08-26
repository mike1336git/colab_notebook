#-----------------------------------------------------------------------------------
#
#  py002_wavePacketQED1D
#    Copyright(C) 2021 Mitsuru Ikeuchi
#    Released under the MIT license ( https://opensource.org/licenses/MIT ) 
#
#    ver 0.0.0  2021.11.18 created, last updated on 2021.11.24
#
#
#--------------------  QED: Quantum Electron Dynamics 1D  ---------------------------
#
# - time dependent Schrodinger equation: i(d/dt)psi(r,t) = H psi(r,t)
# - time evolution
#    psi(r,t+dt) = exp(-i dt H) psi(r,t),  (H:Hamiltonian of the system)
#      H = -delta/2 + V(r), delta = d^2/dx^2
#    psi(r,t+dt) = exp(-i dt H) psi(r,t) nearly=
#      {exp(-i(dt/2)V} {exp(i dt(delta/2)} {exp(-i(dt/2)V} psi(r,t)
# - algorism: {exp(i dt(delta/2)}
#     QED: Watanabe's algorithm (semi-implicit method)
#     Naoki Watanabe, Masaru Tsukada; arXiv:physics/0011068v1
#     (Published from Physical Review E. 62, 2914, (2000).)
#
#     Cayley's form : exp(i dt delta/2) nearly= (1 + i dt delta/4)/(1 - i dt delta/4)
#       psi(r,t+dt) = exp(i dt delta/2) psi(r,dt)
#       (1 - i dt delta/4) psi(r,t+dt) = (1 + i dt delta/4) psi(r,t)
#
#     difference form psi(r,t) --> psi(j,n)
#        psi(j,n+1) - i (dt/dx^2)/4 {psi(j-1,n+1))-2psi(j,n+1)+psi(j+1,n+1)}
#               = psi(j,n) + i (dt/dx^2)/4 {psi(j-1,n))-2psi(j,n)+psi(j+1,n)}
#        x i(4dx^2/dt) by each term
#        psi(j-1,n+1) + A Psi(j,n+1) + psi(j+1,n+1) = -psi(j-1,n) + B Psi(j,n) -psi(j+1,n)
#          where A=(i4dx^2/dt)-2, B=(i4dx^2/dt)+2
#          bnj = -psi(j-1,n) + B Psi(j,n) -psi(j+1,n) is calculated using known psi(j,n)
#          psi(j-1,n+1) + A Psi(j,n+1) + psi(j+1,n+1) = bnj
#
#     solve tri-diagonal equation  A X = B
#         | a1  1  0  0 |  | x1 |    | b1 |
#         |  1 a2  1  0 |  | x2 |  = | b2 |
#         |  0  1 a3  1 |  | x3 |    | b3 |
#         |  0  0  1 a4 |  | x4 |    | b4 |
#
#       u(1) = 1.0/a(1)  // u() : work vector
#       x(1) = b(1)*u(1)
#
#       for(i=2; i<=N-2; i++) { //forward elimination
#          u[i] = 1/(a[i]-u[i-1])
#          x[i] = (b[i]-x[i-1])*u[i]
#       }
#       for(i=N-3; i>=1; i--) { //backward substitution
#          x[i] -= x[i+1]*u[i]
#       }
#
#-----------------------------------------------------------------------------------

import sys, pygame
from pygame.locals import *
import math, random


pygame.init()
pygame.display.set_caption("py002 wave packet QED1D")


#-------------  set global  --------------------------------------------------------

# au: atomic unit hBar=1,e=1,me=1,a0=1
gc_auLength = 5.29177211e-11            # (m) 1(au) = auLength (m)
gc_auTime = 2.418884326e-17             # (s) 1(au) = auTime (s)
gc_auEnergy = 4.35974465e-18            # (J) 1(au) = auEnergy (J)
gc_au2eV = 27.211386                    # (eV) 1(au) = 27.211386 (eV)

g_NNx = 400                             # number of space division (NNx*dx = x-length)
g_dx = 0.5                              # (au) space division 
g_sysTime = 0.0                         # (au) system time
g_timeStep = 0.5*g_dx*g_dx              # (au) time step dt
g_lossSW = 0                            # g_losSW  0:no loss  1:energy loss (steepest descent method)
g_dampingFactor = 0.1                   # for lossEnergy() damping factor :steepest descent method

g_phRe = [ 0.0 for i in range(g_NNx) ]  # real part of wave function
g_phIm = [ 0.0 for i in range(g_NNx) ]  # imaginary part of wave function
g_vv = [ 0.0 for i in range(g_NNx) ]    # external potential
g_bRe = [ 0.0 for i in range(g_NNx) ]   # real part of b vector (b,u: see avove QED1D document)
g_bIm = [ 0.0 for i in range(g_NNx) ]   # imaginary part of b vector
g_uRe = [ 0.0 for i in range(g_NNx) ]   # real part of u vector
g_uIm = [ 0.0 for i in range(g_NNx) ]   # imaginary part of u vector
g_wrkRe = [ 0.0 for i in range(g_NNx) ] # for lossEnergy() real part of work vector
g_wrkIm = [ 0.0 for i in range(g_NNx) ] # for lossEnergy() imaginary part of work vector

# screen surface
g_screenSize = (g_width,g_height) = (440, 500)
g_screen = pygame.display.set_mode(g_screenSize)
g_backgroundColor = (220,200,150)

# box
g_boxPos = (g_xBoxPos,g_yBoxPos) = (20,20)
g_boxSize = (g_boxWidth,g_boxHeight) = (400,400)
g_boxRect = (g_xBoxPos, g_yBoxPos, g_boxWidth, g_boxHeight)
g_boxColor = (150, 150, 150)

# text
g_font = pygame.font.Font(None, 20) # font size 20
g_textColor = (125, 125, 63)

g_theme = 1                             # thema number 0 ... 4
g_themeStr = ('free space', 'parabolic','tunnel effect','step hill','higher hill')
g_inkey = 0                             # like inkey 
g_pauseFlag = 0                         # pauseFlag 0:evolve, 1:pause


#-------------  define function  ---------------------------------------------------

def norm():
	s = 0.0
	for i in range(g_NNx):
		s += (g_phRe[i]*g_phRe[i]+g_phIm[i]*g_phIm[i])*g_dx
	
	return s

def normalize():
	a = math.sqrt(norm())
	for i in range(1,g_NNx-1):
			g_phRe[i] = g_phRe[i]/a
			g_phIm[i] = g_phIm[i]/a

# kinetic energy = <psi(x)|-d^2/dx^2|psi(x)>
def kineticEnergy():
	s = 0.0
	h2 = 2.0*g_dx*g_dx
	for i in range(1,g_NNx-1):
		hhphRe = (2.0*g_phRe[i]-g_phRe[i+1]-g_phRe[i-1])/h2
		hhphIm = (2.0*g_phIm[i]-g_phIm[i+1]-g_phIm[i-1])/h2
		s += (g_phRe[i]*hhphRe + g_phIm[i]*hhphIm)*g_dx
	return s

# potential energy = <psi(x)|V(x)|psi(x)>
def potentialEnergy():
	s = 0.0
	for i in range(1,g_NNx-1):
		s += g_vv[i]*(g_phRe[i]*g_phRe[i] + g_phIm[i]*g_phIm[i])*g_dx
	return s


#------------  set initial condition  ----------------------------------------------

def setFreePotential():
	for i in range(g_NNx):
		g_vv[i] = 0.0

def setParabolicPotential(xpos, k):
	for i in range(g_NNx):
		x = i*g_dx
		g_vv[i] = k*(x-xpos)*(x-xpos)

def setWallPotential(xpos, width, hight):
	for i in range(g_NNx):
		x = i*g_dx
		if x >= xpos and x < xpos+width:
			g_vv[i] = hight
		else:
			g_vv[i] = 0.0

def setWave(wavePos, waveWidth, momentum):
	a = pow(2.0*math.pi*waveWidth*waveWidth,-0.25)
	for i in range(g_NNx):
		x = i*g_dx
		phAb = a*math.exp(-((x-wavePos)/(2.0*waveWidth))*((x-wavePos)/(2.0*waveWidth)))
		phPh = momentum*x
		g_phRe[i] = phAb*math.cos(phPh)
		g_phIm[i] = phAb*math.sin(phPh)
		
	g_phRe[0] = 0.0
	g_phIm[0] = 0.0
	g_phRe[g_NNx-1] = 0.0
	g_phIm[g_NNx-1] = 0.0

	normalize()

def setInitialCondition(theme):
	global g_inkey, g_pauseFlag, g_sysTime, g_theme, g_lossSW

	g_theme = theme
	g_inkey = 0
	g_pauseFlag = 0
	g_sysTime = 0.0
	g_lossSW = 0
	
	# set potential and wave
	xMax = g_NNx*g_dx
	if theme==0:
		# 0:free space
		setFreePotential()
		setWave(0.25*xMax, 10.0, 1.0) # (wavePos,waveWidth,momentum), K=momentum^2/(2me) = 0.5
	elif theme==1:
		# 1:parabolic potential 
		setParabolicPotential(0.5*xMax, 0.0001)  # (xpos,k)
		setWave(0.25*xMax, 10.0, 0.0)  # (wavePos,waveWidth,momentum)
	elif theme==2:
		# 2:tunnel effect
		setWallPotential(0.5*xMax, 10.0*g_dx, 0.55)  # (xpos,width,hight)
		setWave(0.25*xMax, 10.0, 1.0)  # (wavePos,waveWidth,momentum), K=momentum^2/(2me) = 0.5
	elif theme==3:
		# 3:step hill
		setWallPotential(0.5*xMax, 0.5*xMax, 0.3)  # (xpos,width,hight)
		setWave(0.25*xMax, 10.0, 1.0)  # (wavePos,waveWidth,momentum), K=momentum^2/(2me) = 0.5
	elif theme==4:
		# 4:step higher hill
		setWallPotential(0.7*xMax, 0.5*xMax, 0.8)  # (xpos,width,hight)
		setWave(0.25*xMax, 10.0, 1.0)  # (wavePos,waveWidth,momentum), K=momentum^2/(2me) = 0.5


#-------------  time evolution  ----------------------------------------------------

def kxStep(dt):
	bbRe = 2.0
	bbIm = 4*g_dx*g_dx/dt
	aaRe = -2.0
	aaIm = 4*g_dx*g_dx/dt
	aaAb = aaRe*aaRe+aaIm*aaIm

	# b-vector
	for i in range(1,g_NNx-1):
		g_bRe[i] = bbRe*g_phRe[i]-bbIm*g_phIm[i] - g_phRe[i+1] - g_phRe[i-1]
		g_bIm[i] = bbRe*g_phIm[i]+bbIm*g_phRe[i] - g_phIm[i+1] - g_phIm[i-1]

	# u(1) and ph(1)
	g_uRe[1] = aaRe/aaAb
	g_uIm[1] = -aaIm/aaAb
	g_phRe[1] = g_bRe[1]*g_uRe[1] - g_bIm[1]*g_uIm[1]
	g_phIm[1] = g_bIm[1]*g_uRe[1] + g_bRe[1]*g_uIm[1]
	
	# forward elimination
	for i in range(2,g_NNx-1):
		auAb = (aaRe-g_uRe[i-1])*(aaRe-g_uRe[i-1])+(aaIm-g_uIm[i-1])*(aaIm-g_uIm[i-1])
		g_uRe[i] = (aaRe-g_uRe[i-1])/auAb
		g_uIm[i] = -(aaIm-g_uIm[i-1])/auAb
		g_phRe[i] = (g_bRe[i]-g_phRe[i-1])*g_uRe[i] - (g_bIm[i]-g_phIm[i-1])*g_uIm[i]
		g_phIm[i] = (g_bRe[i]-g_phRe[i-1])*g_uIm[i] + (g_bIm[i]-g_phIm[i-1])*g_uRe[i]
	
	# backward substitution
	for i in range(g_NNx-3, 0, -1):
		g_phRe[i] -= g_phRe[i+1]*g_uRe[i] - g_phIm[i+1]*g_uIm[i]
		g_phIm[i] -= g_phRe[i+1]*g_uIm[i] + g_phIm[i+1]*g_uRe[i]

def phaseStep(hdt): # evolve hdt=0.5*dt

	for i in range(1,g_NNx-1):
		th = hdt*g_vv[i]
		cs = math.cos(th)
		sn = math.sin(th)
		phr = g_phRe[i]
		phi = g_phIm[i]
		g_phRe[i] =  cs*phr+sn*phi
		g_phIm[i] = -sn*phr+cs*phi

# steepest descent method: |psi(next)> = |psi> - damp*<psi|H-E|psi>
def lossEnergy(damp):
	h2 = 2.0*g_dx*g_dx
	ee = kineticEnergy() + potentialEnergy()
	
	for i in range(1,g_NNx-1):
		g_wrkRe[i] = -(g_phRe[i+1]+g_phRe[i-1]-2.0*g_phRe[i])/h2+(g_vv[i]-ee)*g_phRe[i]
		g_wrkIm[i] = -(g_phIm[i+1]+g_phIm[i-1]-2.0*g_phIm[i])/h2+(g_vv[i]-ee)*g_phIm[i]
	
	for i in range(1,g_NNx-1):
		g_phRe[i] -= damp*g_wrkRe[i]
		g_phIm[i] -= damp*g_wrkIm[i]
	
	normalize()

def timeEvolution(dt, nCalc, lossSW):
	global g_sysTime
	
	dt = g_timeStep
	
	for i in range(nCalc):
		g_sysTime += dt
		phaseStep(0.5*dt)
		kxStep(dt)
		phaseStep(0.5*dt)

	if lossSW==1 : lossEnergy(g_dampingFactor)


#-------------  draw  --------------------------------------------------------------

def drawText(txt,x,y):
	# render(text, antialias, color, background=None) -> Surface
	text = g_font.render(txt, 1, g_textColor)
	# blit(source, dest, area=None, special_flags=0) -> Rect
	g_screen.blit(text, (x,y))

# draw potential V(x) and probability density |psi(x)|^2
def drawPotentialAndDensity(vmag,pmag):
	# vmag  y = vmag*g_vv         recomended 200
	# pmag  yp = pmag*|psi(x)|^2  recomended 1000
	px = g_xBoxPos
	py = g_yBoxPos+300
	for i in range(g_NNx-1):
		y1 = round(vmag*g_vv[i])
		y2 = round(vmag*g_vv[i+1])
		#line(surface, color, start_pos, end_pos) -> Rect
		#line(surface, color, start_pos, end_pos, width=1) -> Rect
		pygame.draw.line(g_screen, (0,250,0), [i+px, -y1+py], [i+1+px,-y2+py] )
		
		yp1 = round(pmag*(g_phRe[i]*g_phRe[i]+g_phIm[i]*g_phIm[i]))
		if yp1>0:
			pygame.draw.line(g_screen, (250,250,0), [i+px, -y1+py], [i+px,-y1-yp1+py] )

# draw phase at total energy level
def drawWaveAndLevel(vmag,fmag):
	# vmag  y = vmag*g_vv         recomended 200
	# fmag  ph0 = fmag*Re(ph), ph1 = fmag*Im(ph), phaseAngle = atan2(ph1,ph0)  recomended 100
	px = g_xBoxPos
	py = g_yBoxPos+300
	viewTheta = (10.0/180.0)*math.pi
	sinth = math.sin(viewTheta)
	costh = math.cos(viewTheta)
	te = kineticEnergy()+potentialEnergy()
	for i in range(g_NNx-1):
		ph0 = g_phRe[i]*fmag
		ph1 = g_phIm[i]*fmag
		xi = px+(i+sinth*ph1)
		yi = py-(ph0+costh*ph1)-te*vmag
		phaseInDeg = math.atan2(ph1,ph0)*180.0/math.pi
		lineColor = [int(250.0*(phaseInDeg+180.0)/360.0),100,100]
		ph0 = g_phRe[i+1]*fmag
		ph1 = g_phIm[i+1]*fmag
		xip = px+(i+1+sinth*ph1)
		yip = py-(ph0+costh*ph1)-te*vmag
		pygame.draw.line(g_screen, lineColor, [xi,yi], [xip,yip] )

def drawScreen():
	# clear screen
	g_screen.fill(g_backgroundColor)
	
	# draw box
	# rect(surface, color, rect) -> Rect (filled box)
	# draw rect outline: rect(surface, color, rect, width, ... ) -> Rect
	pygame.draw.rect(g_screen, g_boxColor, g_boxRect)

	# draw caption
	yOffset = g_yBoxPos + g_boxHeight + 10
	drawText('box(au) = %s, [0-4]:select theme  [p]:pause/go,  [l] lossSW %s' % \
				( str(round(g_NNx*g_dx,1)), ('off','on')[g_lossSW] ), 20, 3)
	drawText('time(au) = '+str(round(g_sysTime,2)),20,yOffset)
	drawText('norm = '+str(norm()),220,yOffset)
	drawText('Total Energy(au) = '+str(kineticEnergy()+potentialEnergy()), 20, yOffset+20)
	drawText('theme %d : %s ' % (g_theme, g_themeStr[g_theme]), 20, yOffset+40)
	
	# draw potential V(x) and probability density |psi(x)|^2
	# drawPotentialAndDensity(vmag,pmag)
	drawPotentialAndDensity(200,1000)
	
	# draw phase at total energy level
	# drawWaveAndLevel(vmag,fmag)
	drawWaveAndLevel(200,100)
		
	# Update the full display Surface to the screen: flip() -> None
	pygame.display.flip()


#-------------  main  --------------------------------------------------------------

# set potential and wave function  setInitialCondition(theme)
setInitialCondition(1)

while True:
	if g_pauseFlag==0:
		# timeEvolution(dt, nCalc, lossSW)
		timeEvolution(g_timeStep, 4, g_lossSW)
	
	drawScreen()
	
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()
		if event.type == KEYDOWN:
			g_inkey = event.key
			if g_inkey == K_ESCAPE: 
				pygame.quit()
				sys.exit()
	
	if g_inkey == K_p: g_pauseFlag = (g_pauseFlag+1)%2
	elif g_inkey == K_0: setInitialCondition(0) # setInitialCondition(theme)
	elif g_inkey == K_1: setInitialCondition(1)
	elif g_inkey == K_2: setInitialCondition(2)
	elif g_inkey == K_3: setInitialCondition(3)
	elif g_inkey == K_4: setInitialCondition(4)
	elif g_inkey == K_l: g_lossSW = (g_lossSW+1)%2
	g_inkey = 0

	pygame.time.delay(5)

# end

