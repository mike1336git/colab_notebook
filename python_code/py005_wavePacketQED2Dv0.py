#-----------------------------------------------------------------------------------
#
#  py005_wavePacketQED2D
#    Copyright(C) 2021 Mitsuru Ikeuchi
#    Released under the MIT license ( https://opensource.org/licenses/MIT ) 
#
#    ver 0.0.0  2021.11.20 created, last updated on 2021.11.28
#
#
#--------------------  QED: Quantum Electron Dynamics 2D  ---------------------------
#
# - time dependent Schrodinger equation: i(d/dt)psi(r,t) = H psi(r,t)
# - time evolution
#    psi(r,t+dt) = exp(-i dt H) psi(r,t),  (H:Hamiltonian of the system)
#      H = -delta/2 + V(r), delta = d^2/dx^2 + d^2/dy^2
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
pygame.display.set_caption('py005 Quantum Electron Dynamics 2D')


#-------------  set global  --------------------------------------------------------

# au: atomic unit hBar=1,e=1,me=1,a0=1
gc_auLength = 5.29177211e-11            # (m) 1(au) = auLength (m)
gc_auTime = 2.418884326e-17             # (s) 1(au) = auTime (s)
gc_auEnergy = 4.35974465e-18            # (J) 1(au) = auEnergy (J)
gc_au2eV = 27.211386                    # (eV) 1(au) = 27.211386 (eV)
gc_pi = math.pi                         # Pi ~ 3.141592653589792...

g_NNx = 160                             # number of space division (NNx*dx = x-length)
g_NNy = 160                             # number of space division (NNy*dy = y-length)
g_sysTime = 0.0                         # (au) system time
g_dx = 0.5                              # (au) space x-division 
g_dy = 0.5                              # (au) space y-division
g_timeStep = 0.5*g_dx*g_dx              # (au) time step dt
g_lossSW = 0                            # g_losSW  0:no loss  1:energy loss (steepest descent method)
g_dampingFactor = 0.05                  # for lossEnergy() damping factor :steepest descent method

# wave function psi(x,y) --> psi[i][j]  Re(psi(x,y)) = psi[i][j][0], Im(psi(x,y)) = psi[i][j][1]
g_psi = [[[ 0.0, 0.0 ] for j in range(g_NNy)] for i in range(g_NNx)]
g_wrk = [[[ 0.0, 0.0 ] for j in range(g_NNy)] for i in range(g_NNx)] # work wave function in kxStep(),kyStep()
g_vv = [[ 0.0 for j in range(g_NNy) ] for i in range(g_NNx)]  # external potential V(x,y) --> vv[i][j]
g_bRe = [ 0.0 for i in range(g_NNx) ]   # real part of b vector in kxStep(),kyStep()
g_bIm = [ 0.0 for i in range(g_NNx) ]   # imaginal part of b vector in kxStep(),kyStep()
g_uRe = [ 0.0 for i in range(g_NNx) ]   # real part of u vector in kxStep(),kyStep()
g_uIm = [ 0.0 for i in range(g_NNx) ]   # imaginal part of u vector in kxStep(),kyStep()
g_srnd = [ 0.0 for i in range(1002) ]   # 1000 RND orderd series 0 to 1,use drawCloud()
g_cloud = [[ 0 for j in range(g_NNy)] for i in range(g_NNx)] # if cloud[i][j]>0, plot cloud point
g_hue = [[ 0, 0, 0 ] for i in range(360)] # hsl hue color hue[degree] degree: 0,1,2,...,359 see setHueColor()

# screen surface
g_screenSize = (g_width,g_height) = (440, 540)
g_screen = pygame.display.set_mode(g_screenSize)
g_backgroundColor = (220,200,150)

# box
g_boxPos = (g_xBoxPos,g_yBoxPos) = (20,60)
g_boxSize = (g_boxWidth,g_boxHeight) = (400,400)
g_boxRect = (g_xBoxPos, g_yBoxPos, g_boxWidth, g_boxHeight)
g_boxColor = (150, 150, 150)

# text
g_font = pygame.font.Font(None, 20) # font size 20
g_textColor = (125, 125, 63)

g_theme = 1                             # thema number 0 ... 5
g_themeStr = ('free space', 'parabolic','tunnel effect','step hill','double slit','cylinder')
g_inkey = 0                             # like inkey 
g_pauseFlag = 0                         # pauseFlag 0:evolve, 1:pause
g_drawMode = 0                          # drawMode 0:density-2D, 1:along x-axis
g_drawStr = ('density','phase','prob.current','cloud','along x-axis')

#-------------  define function  ---------------------------------------------------

# set HSL hue color  R=g_hue[degree][0],  G=g_hue[degree][1],  B=g_hue[degree][2]
def setHueColor():
	for deg in range(360):
		if (deg<60):
			x = deg
			g_hue[deg][0] = 255
			g_hue[deg][1] = int(255.0*x/60.0)
			g_hue[deg][2] = 0
		elif (deg<120):
			x = deg-60
			g_hue[deg][0] = int(255.0*(60.0-x)/60.0)
			g_hue[deg][1] = 255
			g_hue[deg][2] = 0
		elif (deg<180):
			x = deg-120
			g_hue[deg][0] = 0
			g_hue[deg][1] = 255
			g_hue[deg][2] = int(255.0*x/60.0)
		elif (deg<240):
			x = deg-180
			g_hue[deg][0] = 0
			g_hue[deg][1] = int(255.0*(60.0-x)/60.0)
			g_hue[deg][2] = 255
		elif (deg<360):
			x = deg-240
			g_hue[deg][0] = int(255.0*x/120.0)
			g_hue[deg][1] = 0
			g_hue[deg][2] = int(255.0*(120.0-x)/120.0)
	#for i in range(360):
	#	print('deg:%3d R:%3d  G:%3d  B%3d' % (i,g_hue[i][0],g_hue[i][1],g_hue[i][2]))

def norm(ph):
	s = 0.0
	for i in range(1,g_NNx-1):
		for j in range(1,g_NNy-1):
			s += (ph[i][j][0]*ph[i][j][0]+ph[i][j][1]*ph[i][j][1])
	return s*g_dx*g_dy

def normalize(ph):
	a = 1.0/math.sqrt(norm(ph))
	for i in range(1,g_NNx-1):
		for j in range(1,g_NNy-1):
			ph[i][j][0] *= a
			ph[i][j][1] *= a

# potential energy = <ph|V(x,y)|ph>
def meanPotential(ph, vv):
	p = 0.0
	for i in range(1,g_NNx-1):
		for j in range(1,g_NNy-1):
			p += vv[i][j]*(ph[i][j][0]*ph[i][j][0]+ph[i][j][1]*ph[i][j][1])
	return p*g_dx*g_dy

# mean kinetic energy = <ph| -(1/2)(d^2/dx^2 + d^2/dy^2) |ph>
def meanKinetic(ph):
	h2 = g_dx*g_dx
	p = 0.0
	for i in range(1,g_NNx-1):
		for j in range(1,g_NNy-1):
			d2phRe = (ph[i+1][j][0]+ph[i-1][j][0]+ph[i][j+1][0]+ph[i][j-1][0]-4.0*ph[i][j][0])/h2
			d2phIm = (ph[i+1][j][1]+ph[i-1][j][1]+ph[i][j+1][1]+ph[i][j-1][1]-4.0*ph[i][j][1])/h2
			p += (ph[i][j][0]*d2phRe+ph[i][j][1]*d2phIm)
	return -0.5*p*g_dx*g_dy

def psiDensity(i,j):
	return (g_psi[i][j][0]*g_psi[i][j][0]+g_psi[i][j][1]*g_psi[i][j][1])

def psiPhase(i,j):  # 0 ... 2 pi
	return (math.pi + math.atan2(g_psi[i][j][1],g_psi[i][j][0]))  # atan2(y,x)

def psiXCurrent(i,j):
	pRe = (g_psi[i+1][j][1]-g_psi[i-1][j][1])/(2*g_dx)
	pIm = (-g_psi[i+1][j][0]+g_psi[i-1][j][0])/(2*g_dx)
	return (g_psi[i][j][0]*pRe + g_psi[i][j][1]*pIm)*g_dx*g_dy

def psiYCurrent(i,j):
	pRe = (g_psi[i][j+1][1]-g_psi[i][j-1][1])/(2*g_dy)
	pIm = (-g_psi[i][j+1][0]+g_psi[i][j-1][0])/(2*g_dy)
	return (g_psi[i][j][0]*pRe + g_psi[i][j][1]*pIm)*g_dx*g_dy


#------------  set initial condition  ----------------------------------------------

def setGaussianWave(xPos,yPos,waveWidth,kx,ky):
	for i in range(1,g_NNx-1):
		for j in range(1,g_NNy-1):
			x = i*g_dx
			y = j*g_dy
			phAb = math.exp(-((x-xPos)*(x-xPos)+(y-yPos)*(y-yPos))/(4*waveWidth*waveWidth) )
			phPh = kx*x+ky*y
			g_psi[i][j][0] = phAb*math.cos(phPh)
			g_psi[i][j][1] = phAb*math.sin(phPh)
	for i in range(g_NNx):
		g_psi[i][0][0] = 0.0
		g_psi[i][0][1] = 0.0
		g_psi[i][g_NNy-1][0] = 0.0
		g_psi[i][g_NNy-1][1] = 0.0
	for j in range(g_NNy):
		g_psi[0][j][0] = 0.0
		g_psi[0][j][1] = 0.0
		g_psi[g_NNx-1][j][0] = 0.0
		g_psi[g_NNx-1][j][1] = 0.0
	normalize(g_psi);

# V(r)= k0*r^2
def setParabolicPotential(k0):
	aa = k0/(g_NNx*g_dx*g_NNx*g_dx/4.0)
	x0 = g_NNx*g_dx/2.0
	y0 = g_NNy*g_dy/2.0
	for i in range(g_NNx):
		x = i*g_dx
		for j in range(g_NNy):
			y = j*g_dy
			g_vv[i][j] = aa*((x-x0)*(x-x0)+(y-y0)*(y-y0))

def setWallPotential(xPos, vThick, vHeight):
	for i in range(g_NNx):
		x = i*g_dx
		for j in range(g_NNy):
			y = j*g_dy
			if (x>=xPos and x<xPos+vThick):
				g_vv[i][j] = vHeight
			else:
				g_vv[i][j] = 0.0

def setCylinderPotential(xPos,yPos, radius, vHeight):
	for i in range(g_NNx):
		x = i*g_dx - xPos
		for j in range(g_NNy):
			y = j*g_dy - yPos
			if (x*x+y*y < radius*radius):
				g_vv[i][j] = vHeight
			else:
				g_vv[i][j] = 0.0

def setSlitPotential(wallPos, wallThick, wallHeight, slitWidth, slitSpan):
	ym = g_NNy*g_dy/2.0
	w = slitWidth/2.0
	d = slitSpan/2.0
	for i in range(g_NNx):
		x = i*g_dx
		for j in range(g_NNy):
			y = j*g_dy
			if (x>=wallPos and x<wallPos+wallThick):
				g_vv[i][j] = wallHeight
				if ((y>=ym-d-w and y<=ym-d+w) or (y>=ym+d-w and y<=ym+d+w)):
					g_vv[i][j] = 0.0
			else:
				g_vv[i][j] = 0.0


def setInitialCondition(theme):
	global g_inkey, g_pauseFlag, g_sysTime, g_theme, g_lossSW

	g_theme = theme
	g_inkey = 0
	g_pauseFlag = 0
	g_sysTime = 0.0
	g_lossSW = 0
	
	# set potential and wave
	xMax = g_NNx*g_dx
	yMax = g_NNy*g_dy
	if (theme==0):   # 0:free space
		setGaussianWave(0.25*xMax,0.5*yMax,5.0,1.0,0.0) # (xPos,yPos,waveWidth,kx,ky)
		setWallPotential(0.5*xMax, 5*g_dx, 0.0) # (xPos, vThick, vHeight)
	elif (theme==1): # 1:parabolic potential
		setGaussianWave(xMax/2.0,yMax/4.0,3.0,1.0,0.0) # (xPos,yPos,waveWidth,kx,ky)
		setParabolicPotential(2.0) # v(x)=k0*(x-x0)^2, k0=2.0
	elif (theme==2): # 2:tunnel effect
		setGaussianWave(0.25*xMax,0.5*yMax,5.0,1.0,0.0) # (xPos,yPos,waveWidth,kx,ky) K=0.5
		setWallPotential(0.5*xMax, 5*g_dx, 0.55) # (xPos, vThick, vHeight) 
	elif (theme==3): # 3:step hill
		setGaussianWave(0.25*xMax,0.25*yMax,5.0,1.0,1.0) # (xPos,yPos,waveWidth,kx,ky)
		setWallPotential(0.5*xMax, xMax, 0.4) # (xPos, vThick, vHeight)
	elif (theme==4): # 4:double slit
		setGaussianWave(0.25*xMax,0.5*yMax,5.0,1.0,0.0) # (xPos,yPos,waveWidth,kx,ky)
		setSlitPotential(0.5*xMax,5*g_dx,4.0,0.05*yMax,0.10*yMax) # (Pos,Thick,Height,slitWidth,slitSpan)
	elif (theme==5): # 5:cylinder
		setGaussianWave(0.25*xMax,0.5*yMax,5.0,1.0,0.0) # (xPos,yPos,waveWidth,kx,ky)
		setCylinderPotential(0.6*xMax,0.4*yMax,0.1*xMax,1.0) # (xPos,yPos, radius, vHeight)


#-------------  time evolution  ----------------------------------------------------

def phaseStep(ph, vv, dt):
	for i in range(1,g_NNx-1):
		for j in range(1,g_NNy-1):
			th = dt*g_vv[i][j]
			cs = math.cos(th)
			sn = math.sin(th)
			phr = ph[i][j][0]
			phi = ph[i][j][1]
			ph[i][j][0] = cs*phr+sn*phi
			ph[i][j][1] = cs*phi-sn*phr

def kxStep(ph, dt):
	a = 4.0*g_dy*g_dy/dt;
	aaAb = 4.0+a*a;
	for j in range(1,g_NNy-1):
		# set b[], u[1],ph[1][]
		for i in range(1,g_NNx-1):
			g_bRe[i] = 2.0*ph[i][j][0]-a*ph[i][j][1] - ph[i+1][j][0] - ph[i-1][j][0]
			g_bIm[i] = 2.0*ph[i][j][1]+a*ph[i][j][0] - ph[i+1][j][1] - ph[i-1][j][1]
		g_uRe[1] = -2.0/aaAb
		g_uIm[1] = -a/aaAb
		ph[1][j][0] = g_bRe[1]*g_uRe[1] - g_bIm[1]*g_uIm[1]
		ph[1][j][1] = g_bIm[1]*g_uRe[1] + g_bRe[1]*g_uIm[1]
		# forward elimination
		for i in range(2,g_NNx-1):
			auAb = (-2.0-g_uRe[i-1])*(-2.0-g_uRe[i-1])+(a-g_uIm[i-1])*(a-g_uIm[i-1])
			g_uRe[i] = (-2.0-g_uRe[i-1])/auAb
			g_uIm[i] = -(a-g_uIm[i-1])/auAb
			ph[i][j][0] = (g_bRe[i]-ph[i-1][j][0])*g_uRe[i] - (g_bIm[i]-ph[i-1][j][1])*g_uIm[i]
			ph[i][j][1] = (g_bRe[i]-ph[i-1][j][0])*g_uIm[i] + (g_bIm[i]-ph[i-1][j][1])*g_uRe[i]
		#backward substitution
		for i in range(g_NNx-3, 0, -1):
			ph[i][j][0] -= ph[i+1][j][0]*g_uRe[i] - ph[i+1][j][1]*g_uIm[i]
			ph[i][j][1] -= ph[i+1][j][0]*g_uIm[i] + ph[i+1][j][1]*g_uRe[i]

def kyStep(ph, dt):
	a = 4.0*g_dy*g_dy/dt;
	aaAb = 4.0+a*a;
	for i in range(1,g_NNx-1):
		# set b[], u[1],ph[][1]
		for j in range(1,g_NNy-1):
			g_bRe[j] = 2.0*ph[i][j][0]-a*ph[i][j][1] - ph[i][j+1][0] - ph[i][j-1][0]
			g_bIm[j] = 2.0*ph[i][j][1]+a*ph[i][j][0] - ph[i][j+1][1] - ph[i][j-1][1]
		g_uRe[1] = -2.0/aaAb
		g_uIm[1] = -a/aaAb
		ph[i][1][0] = g_bRe[1]*g_uRe[1] - g_bIm[1]*g_uIm[1]
		ph[i][1][1] = g_bIm[1]*g_uRe[1] + g_bRe[1]*g_uIm[1]
		# forward elimination
		for j in range(2,g_NNy-1):
			auAb = (-2.0-g_uRe[j-1])*(-2.0-g_uRe[j-1])+(a-g_uIm[j-1])*(a-g_uIm[j-1])
			g_uRe[j] = (-2.0-g_uRe[j-1])/auAb
			g_uIm[j] = -(a-g_uIm[j-1])/auAb
			ph[i][j][0] = (g_bRe[j]-ph[i][j-1][0])*g_uRe[j] - (g_bIm[j]-ph[i][j-1][1])*g_uIm[j]
			ph[i][j][1] = (g_bRe[j]-ph[i][j-1][0])*g_uIm[j] + (g_bIm[j]-ph[i][j-1][1])*g_uRe[j]
		# backward substitution
		for j in range(g_NNy-3, 0, -1):
			ph[i][j][0] -= ph[i][j+1][0]*g_uRe[j] - ph[i][j+1][1]*g_uIm[j]
			ph[i][j][1] -= ph[i][j+1][0]*g_uIm[j] + ph[i][j+1][1]*g_uRe[j]

# steepest descent method: psi_next = |psi) - damp*|psi)(psi|H-E|psi)
# H = d^2/dx^2 + d^2/dy^2 + V
def lossEnergy(ph, vv, damp):
	h2 = 2.0*g_dx*g_dx;
	ee = meanKinetic(ph) + meanPotential(ph,vv);
	for i in range(1,g_NNx-1):
		for j in range(1,g_NNy-1):
			g_wrk[i][j][0] = -(ph[i+1][j][0]+ph[i-1][j][0]+ph[i][j+1][0]+ph[i][j-1][0]-4.0*ph[i][j][0])/h2 \
								+(g_vv[i][j]-ee)*ph[i][j][0]
			g_wrk[i][j][1] = -(ph[i+1][j][1]+ph[i-1][j][1]+ph[i][j+1][1]+ph[i][j-1][1]-4.0*ph[i][j][1])/h2 \
								+(g_vv[i][j]-ee)*ph[i][j][1]
	for i in range(1,g_NNx-1):
		for j in range(1,g_NNy-1):
			ph[i][j][0] -= damp*g_wrk[i][j][0]
			ph[i][j][1] -= damp*g_wrk[i][j][1]

	normalize(ph)

def timeEvolution(nCalc, lossSW):
	# nCalc:  repeat evolve times 
	# lossSW:  0:no loss, 1:loss
	global g_sysTime

	for i in range(nCalc):
		g_sysTime += g_timeStep
		phaseStep(g_psi,g_vv,0.5*g_timeStep)
		kxStep(g_psi,g_timeStep)
		kyStep(g_psi,g_timeStep)
		phaseStep(g_psi,g_vv,0.5*g_timeStep)
	if (lossSW==1): lossEnergy(g_psi,g_vv,g_dampingFactor)


#-------------  draw  --------------------------------------------------------------

def drawText(txt,x,y):
	# render(text, antialias, color, background=None) -> Surface
	text = g_font.render(txt, 1, g_textColor)
	# blit(source, dest, area=None, special_flags=0) -> Rect
	g_screen.blit(text, (x,y))

def drawPsiDensity(vmag,pmag):
	xPos = g_xBoxPos + 40
	yPos = g_yBoxPos + 40
	for i in range(0,g_NNx-1):
		for j in range(0,g_NNy-1):
			green = round(vmag*g_vv[i][j])
			if green>255: green = 255
			red = round(pmag*psiDensity(i,j))
			if red>255: red = 255
			blue = red
			pygame.draw.rect(g_screen, (red,green,blue), (2*i+xPos,2*j+yPos,2,2))

def drawPsiPhase(vmag,pmag):
	xPos = g_xBoxPos + 40
	yPos = g_yBoxPos + 40
	for i in range(0,g_NNx-1):
		for j in range(0,g_NNy-1):
			ap = pmag*psiDensity(i,j)
			p = psiPhase(i,j)*180.0/gc_pi # p:0.0 ... 360.0
			th = (int(p) + 360) % 360
			v = g_vv[i][j]
			green = round(vmag*v)
			red = 0
			blue = 0
			if (ap>20.0):
				red = round(ap*(g_hue[th][0]/256.0))
				green = round(vmag*v + ap*(g_hue[th][1]/256.0))
				blue = round(ap*(g_hue[th][2]/256.0))
			if red>255: red = 255
			if green>255: green = 255
			if blue>255: blue = 255
			pygame.draw.rect(g_screen, (red,green,blue), (2*i+xPos,2*j+yPos,2,2))

def drawprobCurrent(pmag,velocMag):
	xPos = g_xBoxPos + 40
	yPos = g_yBoxPos + 40
	# velocMag = 300000.0
	for i in range(2,g_NNx-2,4):
		for j in range(2,g_NNy-2,4):
			ap = pmag*psiDensity(i,j)
			if (ap>20.0):
				x1 = 2*i+xPos
				y1 = 2*j+yPos
				x2 = round(x1+psiXCurrent(i,j)*g_timeStep*velocMag)
				y2 = round(y1+psiYCurrent(i,j)*g_timeStep*velocMag)
				if (x2-x1)>0:
					col = (250,80,80)
				else:
					col = (80,80,250)
				pygame.draw.line(g_screen, col, [x1, y1], [x2,y2])

def drawCloud(vmag):
	xPos = g_xBoxPos + 40
	yPos = g_yBoxPos + 40
	if (g_pauseFlag==0):
		set_srnd()
		setCloud()
	for i in range(0,g_NNx-1):
		for j in range(0,g_NNy-1):
			# draw V
			v = g_vv[i][j]
			green = round(vmag*v)
			if green>255: green = 255
			red = 0
			blue = 0
			pygame.draw.rect(g_screen, (red,green,blue), (2*i+xPos,2*j+yPos,2,2))
			# draw cloud
			if (g_cloud[i][j]>0):
				col = (255,80,160)
				if (g_cloud[i][j]==1): col = (140,140,0)
				elif (g_cloud[i][j]==2): col = (200,120,0)
				elif (g_cloud[i][j]==3): col = (240,100,100)
				pygame.draw.rect(g_screen, col, (2*i+xPos,2*j+yPos,2,2))

def setCloud():
	s = 0
	ip = 0
	for i in range(1,g_NNx-1):
		for j in range(1,g_NNy-1):
			g_cloud[i][j] = 0
			s += psiDensity(i,j)*g_dx*g_dy
			while (s>g_srnd[ip] and ip<1000):
				g_cloud[i][j] += 1
				ip += 1

def set_srnd():
	g_srnd[0] = random.random();
	for i in range(1,1001):
		g_srnd[i] = g_srnd[i-1] + random.random()
	for i in range(1000):
		g_srnd[i] = g_srnd[i]/g_srnd[1000]

def drawPsiDensity1D(j,vmag,pmag):
	xPos = g_xBoxPos + 40
	yPos = g_yBoxPos + 300
	# vmag = 100   draw magnitude
	# pmag = 10000 draw magnitude

	# draw potential V(x) and probability density |psi(x)|^2
	for i in range(g_NNx-1):
		y1 = round(vmag*g_vv[i][j])
		y2 = round(vmag*g_vv[i+1][j])
		#line(surface, color, start_pos, end_pos) -> Rect
		#line(surface, color, start_pos, end_pos, width=1) -> Rect
		pygame.draw.line(g_screen, (0,250,0), [2*i+xPos, -y1+yPos], [2*(i+1)+xPos,-y2+yPos] )
		
		yp1 = round(pmag*psiDensity(i,j))
		if yp1>0:
			pygame.draw.line(g_screen, (250,250,0), [2*i+xPos, -y1+yPos], [2*i+xPos,-y1-yp1+yPos],2)

def drawScreen(drawMode):
	# clear screen
	g_screen.fill(g_backgroundColor)
	
	# draw box
	# rect(surface, color, rect) -> Rect (filled box)
	# draw rect outline: rect(surface, color, rect, width, ... ) -> Rect
	pygame.draw.rect(g_screen, g_boxColor, g_boxRect)
	drawText('box(au) = %s x %s' % ( str(round(g_NNx*g_dx,1)),str(round(g_NNy*g_dy,1)) ),g_xBoxPos+40,g_yBoxPos+20)
	
	# draw caption
	yOffset = g_yBoxPos + g_boxHeight + 10
	kk = meanKinetic(g_psi)
	uu = meanPotential(g_psi,g_vv)
	drawText('key menu   [0]:free space, [1]:parabolic, [2]:tunnel effect,', 20, 3)
	drawText('[3]:step hill, [4]:double slit, [5]:cylinder collision', 20, 23)
	drawText('[esc]:quit, [p]:pause/go, [l]:lossSW %d, [d]:change draw mode ' % ( g_lossSW ), 20, 43)
	
	drawText('time(au) = %0.2f' % (g_sysTime),20,yOffset)
	drawText('norm = '+str(norm(g_psi)),220,yOffset)
	drawText('Total Energy(au) = %9.6f' % (kk+uu), 20, yOffset+20)
	drawText('( K = %9.6f  U = %9.6f )' % (kk, uu), 220, yOffset+20)
	drawText('theme %d : %s ' % (g_theme, g_themeStr[g_theme]), 20, yOffset+40)
	drawText('draw mode %d : %s ' % (g_drawMode, g_drawStr[g_drawMode]), 220, yOffset+40)
	
	if drawMode==0: # 0:density
		drawPsiDensity(100,50000) # (vmag,pmag)
	elif drawMode==1: # 1:phase
		drawPsiPhase(100,50000) # (vmag,pmag)
	elif drawMode==2: # 2:prob.current
		drawPsiDensity(100,50000) # (vmag,pmag)
		drawprobCurrent(50000,300000) # (pmag,velocMag)
	elif drawMode==3: # 3:cloud
		drawCloud(100) # (vmag)
	elif drawMode==4: # 4:along x-axis
		drawPsiDensity1D(g_NNy//2,100,10000)
	
	# Update the full display Surface to the screen: flip() -> None
	pygame.display.flip()


#-------------  main  --------------------------------------------------------------

setHueColor()
# set potential and wave function  setInitialCondition(theme)
setInitialCondition(1)

while True:
	if g_pauseFlag==0:
		# timeEvolution(nCalc, lossSW)
		timeEvolution(1, g_lossSW)
	
	drawScreen(g_drawMode)
	
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()
		if event.type == KEYDOWN:
			g_inkey = event.key
			if g_inkey == K_ESCAPE: 
				pygame.quit()
				sys.exit()
	
	# key menu
	if g_inkey == K_p:
		g_inkey = 0
		g_pauseFlag = (g_pauseFlag+1)%2 # pause/go
	elif g_inkey == K_0:
		g_inkey = 0
		setInitialCondition(0) # 0:free space
	elif g_inkey == K_1:
		g_inkey = 0
		setInitialCondition(1) # 1:parabolic potential
	elif g_inkey == K_2:
		g_inkey = 0
		setInitialCondition(2) # 2:tunnel effect
	elif g_inkey == K_3:
		g_inkey = 0
		setInitialCondition(3) # 3:step hill
	elif g_inkey == K_4:
		g_inkey = 0
		setInitialCondition(4) # 4:double slit
	elif g_inkey == K_5:
		g_inkey = 0
		setInitialCondition(5) # 5:cylinder
	elif g_inkey == K_l:
		g_inkey = 0
		g_lossSW = (g_lossSW+1)%2 # loss SW on/off
	elif g_inkey == K_d:
		g_inkey = 0
		g_drawMode = (g_drawMode+1)%4 # change g_drawMode (0-3) exclude 4:drawPsiDensity1D()

	#pygame.time.delay(5)

# end

