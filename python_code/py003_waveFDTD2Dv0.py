#-----------------------------------------------------------------------------------
#
#  py003_waveFDTD2D
#    Copyright(C) 2021 Mitsuru Ikeuchi
#    Released under the MIT license ( https://opensource.org/licenses/MIT ) 
#
#    ver 0.0.0  2021.11.19 created, last updated on 2021.11.23
#
#
#--------------------  FDTD2D: finite-difference time-domain method 2D  -------------
#
# - electro-magnetic field : Maxwell's equations
#     rot H = eps dE/dt + sgm E
#     rot E = -mue dH/dt
#
#   in 2D system: Ez,Hx,Hy - TMz system
#     dEz/dt = (1/eps) (dHy/dx-dHx/dy) - (sgm/eps) Ez
#     dHx/dt = -(1/mue) (dEz/dy-dEy/dz)
#     dHy/dt = -(1/mue) (dEx/dz-dEz/dx)
#
#     dEz = dt(1/eps) (dHy/dx - dHx/dy - sgm Ez)
#     dHx = -dt(1/mue) dEz/dy
#     dHy = dt(1/mue) dEz/dx
#
# - FDTD (Finite Difference Time Domain method)
#     finite difference dA/dx ~> {A(x+h)-A(x)}/h, dA/dt -> {A(t+dt)-A(t)}/dt
#
#     Ez(i,j,t+dt) = Ez(i,j) + dt(1/eps){(Hy(i+1,j)-Hy(i,j))/dx-(Hx(i,j+1)-Hx(i,j))/dy - sgm Ez(i,j)}
#     Hx(i,j,t+dt) = Hx(i,j) - dt(1/mue)(Ez(i,j)-Ez(i,j-1))/dy
#     Hy(i,j,t+dt) = Hy(i,j) + dt(1/mue)(Ez(i,j)-Ez(i-1,j))/dx
#
# - boundary : no-reflect (in vacuum) condition case
#     MUR Hx(n+1,i,0) = Hx(n,i,1)-(dx-dt)/(dx+dt){Hx(n+1,i,1)-Hx(n,i,0)}
#
#     Hx(n)[i][0] = c*Hx(n)[i][0] + Hx(n)[i][1]; c=(dx-dt)/(dx+dt)
#       ...
#     (time evolution : Hx(n)[i][0] no change, Hx(n)[i][1] -> Hx(n+1)[i][1])
#       ...
#     Hx(n+1)[i][0] = Hx(n)[i][0] - c*Hx(n+1)[i][1]
#                   = c*Hx(n)[i][0] + Hx(n)[i][1] - c*Hx(n+1)[i][1]
#                   = Hx(n)[i][1] -c*(Hx(n+1)[i][1] - Hx(n)[i][0])
#
#-----------------------------------------------------------------------------------

import sys, pygame
from pygame.locals import *
import math, random


pygame.init()
pygame.display.set_caption("py003 Ez-Hxy wave FDTD 2D")


#-------------  set global  --------------------------------------------------------

g_NNx = 240                       # field array x-max
g_NNy = 240                       # field array y-max
g_dx = 1.0                        # space division dx =1.0, (dy = 1.0 (no use))
g_sysTime = 0.0                   # system time
g_dt = 1.0                        # time step
g_omega = math.pi/16.0            # generator wave angular velocity: phase angle += omega*dt
g_theta = 0.0                     # generator phase angle

g_Ez = [[0.0 for j in range(g_NNy+1)] for i in range(g_NNx+1)]	# z-component of electric field
g_Hx = [[0.0 for j in range(g_NNy+1)] for i in range(g_NNx+1)]	# x-component of magnetic field
g_Hy = [[0.0 for j in range(g_NNy+1)] for i in range(g_NNx+1)]	# y-component of magnetic field
g_ep = [[0.0 for j in range(g_NNy+1)] for i in range(g_NNx+1)]	# dielectric constant
g_mu = [[0.0 for j in range(g_NNy+1)] for i in range(g_NNx+1)]	# magnetic permeability
g_sg = [[0.0 for j in range(g_NNy+1)] for i in range(g_NNx+1)]	# Electrical conductivity

# screen surface
g_screenSize = (g_width,g_height) = (440, 500)
g_screen = pygame.display.set_mode(g_screenSize)
g_backgroundColor = (220,200,150)

# box
g_boxPos = (g_xBoxPos,g_yBoxPos) = (20,40)
g_boxSize = (g_boxWidth,g_boxHeight) = (400,400)
g_boxRect = (g_xBoxPos, g_yBoxPos, g_boxWidth, g_boxHeight)
g_boxColor = (150, 150, 150)

# text
g_font = pygame.font.Font(None, 20) # font size 20
g_textColor = (125, 125, 63)

g_theme = 1                             # thema number 0 ... 3
g_themeStr = ('free space', 'glass(n=2.0)', 'metal', 'absober')
g_drawMode = 0                          # drawMode 0:Ez-wave 1:Hxy-wave
g_inkey = 0                             # like inkey 
g_pauseFlag = 0                         # pauseFlag 0:evolve, 1:pause
g_nWaves = 2                            # generate number of waves       


#-------------  define function  ---------------------------------------------------

#------------  set initial condition 

def clearField():
	for i in range(g_NNx):
		for j in range(g_NNy):
			g_Ez[i][j] = 0.0
			g_Hx[i][j] = 0.0
			g_Hy[i][j] = 0.0
			g_ep[i][j] = 1.0
			g_mu[i][j] = 1.0
			g_sg[i][j] = 0.0

def setOpticalDevice(eps,mue,sgm):
	for i in range(g_NNx):
		for j in range(g_NNy):
			if (i>=50 and i<120 and j>=50 and j<150):
				g_ep[i][j] = eps
				g_mu[i][j] = mue
				g_sg[i][j] = sgm


def setInitialCondition(theme):
	global g_theme, g_inkey, g_pauseFlag, g_sysTime, g_theta

	g_theme = theme
	g_inkey = 0
	g_pauseFlag = 0
	g_sysTime = 0.0
	g_theta = 0.0

	clearField()
	if (theme==0):  # free space
		eps=1.0
		mue=1.0
		sgm=0.0
	elif (theme==1): # glass n=2.0
		eps=4.0
		mue=1.0
		sgm=0.0
	elif (theme==2): # metal
		eps=1000.0
		mue=1.0
		sgm=1.0
	elif (theme==3): # absorber
		eps=1.01
		mue=1.0
		sgm=0.1

	setOpticalDevice(eps,mue,sgm)


#-------------  time evolution  ----------------------------------------------------

def generateEz(nwave): # plane wave 
	global g_theta

	g_theta += g_omega*0.5*g_dt
	Ezt = math.sin(g_theta)
	a = 0.0
	if (g_theta<2.0*math.pi*nwave):
		a = 1.0
	elif (g_theta<2.0*math.pi*nwave+0.10*math.pi):
		a = math.cos(g_theta)
	if (g_theta<2.0*math.pi*nwave+0.5*math.pi):
		for j in range(g_NNy):
			g_Ez[0][j] = a*a*Ezt

def evolveEz(): # dD/dt=rotH + J , D=eps*E
	dtv2 = 0.5*g_dt;
	for i in range(g_NNx):
		for j in range(g_NNy):
			if (g_ep[i][j]<1000.0): # non-metal
				g_Ez[i][j] += (dtv2/g_ep[i][j])*( (g_Hy[i+1][j]-g_Hy[i][j]) - (g_Hx[i][j+1]-g_Hx[i][j]) \
								- g_sg[i][j]*g_Ez[i][j] )


def evolveHxHy(): # dB/dt=-rotE , B=mue*H
	dtv2 = 0.5*g_dt;

	# boundary : no-reflect (in vacuum) condition (Mur 1st)
	# MUR Hx(n+1,i,0) = Hx(n,i,1)-(dx-dt)/(dx+dt){Hx(n+1,i,1)-Hx(n,i,0)}
	c = (g_dx-dtv2)/(g_dx+dtv2)
	for i in range(g_NNx):
		g_Hx[i][0] = c*g_Hx[i][0] + g_Hx[i][1]
	for i in range(g_NNx-1):
		g_Hx[i][g_NNy] = c*g_Hx[i][g_NNy] + g_Hx[i][g_NNy-1]
	for j in range(g_NNy):
		g_Hy[0][j] = c*g_Hy[0][j] + g_Hy[1][j]
	for j in range(g_NNy):
		g_Hy[g_NNx][j] = c*g_Hy[g_NNx][j] + g_Hy[g_NNx-1][j]

	# Hx(i,j) - dt(1/mue)(Ez(i,j)-Ez(i,j-1))/dy
	for i in range(g_NNx):
		for j in range(1,g_NNy):
			g_Hx[i][j] -= dtv2/g_mu[i][j]*(g_Ez[i][j]-g_Ez[i][j-1])

	# Hy(i,j,t+dt) = Hy(i,j) + dt(1/mue)(Ez(i,j)-Ez(i-1,j))/dx
	for i in range(1,g_NNx):
		for j in range(g_NNy):
			g_Hy[i][j] += dtv2/g_mu[i][j]*(g_Ez[i][j]-g_Ez[i-1][j])

	# boundary : no-reflect (in vacuum) condition (Mur 1st)
	for i in range(g_NNx):
		g_Hx[i][0] -= c*g_Hx[i][1]
	for i in range(g_NNx):
		g_Hx[i][g_NNy] -= c*g_Hx[i][g_NNy-1]
	for j in range(g_NNy):
		g_Hy[0][j] -= c*g_Hy[1][j]
	for j in range(g_NNy):
		g_Hy[g_NNx][j] -= c*g_Hy[g_NNx-1][j]

def evolveField(nCalc,nWaves):
	global g_sysTime
	n=2*nCalc
	g_sysTime += nCalc*g_dt
	for i in range(n): # evolve 0.5dt
		evolveEz()
		generateEz(nWaves)
		evolveHxHy()


#-------------  draw  --------------------------------------------------------------

def drawText(txt,x,y):
	# render(text, antialias, color, background=None) -> Surface
	text = g_font.render(txt, 1, g_textColor)
	# blit(source, dest, area=None, special_flags=0) -> Rect
	g_screen.blit(text, (x,y))

def drawEzWave():
	xPos = g_xBoxPos + 60 # draw x-pos
	yPos = g_yBoxPos + 60 # draw y-pos
	for i in range(0,g_NNx-2,2):
		for j in range(0,g_NNy-2,2):
			Ez = (g_Ez[i][j]+g_Ez[i+1][j]+g_Ez[i][j+1]+g_Ez[i][j+1])/4.0
			red = 0
			green = 0
			blue = 0
			if (Ez>0):
				red = round(Ez*200.0)
				if (red>255): red = 255
			if (Ez<0):
				blue = round(-Ez*200.0)
				if (blue>255): blue = 255
			ep = (g_ep[i][j]+g_ep[i+1][j]+g_ep[i][j+1]+g_ep[i][j+1])/4.0
			if (ep>1.0):
				green = 50
			pygame.draw.rect(g_screen, (red,green,blue), (i+xPos,j+yPos,2,2))	

def drawEzWaveRes1():
	xPos = g_xBoxPos + 60 # draw x-pos
	yPos = g_yBoxPos + 60 # draw y-pos
	pixarry = pygame.PixelArray(g_screen)
	for i in range(g_NNx):
		for j in range(g_NNy):
			Ez = g_Ez[i][j]
			red = 0
			green = 0
			blue = 0
			if (Ez>0):
				red = round(Ez*200.0)
				if (red>255): red = 255
			if (Ez<0):
				blue = round(-Ez*200.0)
				if (blue>255): blue = 255
			ep = g_ep[i][j]
			if (ep>1.0):
				green = 50
			pixarry[i+xPos][j+yPos] = (red,green,blue)
	del pixarry

def drawHxHyWave(mag):
	xPos = g_xBoxPos + 60 # draw x-pos
	yPos = g_yBoxPos + 60 # draw y-pos
	# draw material(ep>1.0)
	for i in range(0,g_NNx-2,2):
		for j in range(0,g_NNy-2,2):
			green = 0
			ep = (g_ep[i][j]+g_ep[i+1][j]+g_ep[i][j+1]+g_ep[i][j+1])/4.0
			if (ep>1.0):
				green = 50
			pygame.draw.rect(g_screen, (0,green,0), (i+xPos,j+yPos,2,2))
	# draw HxHy line
	for i in range(1,g_NNx-2,4):
		for j in range(1,g_NNy-2,4):
			Hx = (g_Hx[i][j]+g_Hx[i+1][j]+g_Hx[i][j+1]+g_Hx[i][j+1])/4.0
			Hy = (g_Hy[i][j]+g_Hy[i+1][j]+g_Hy[i][j+1]+g_Hy[i][j+1])/4.0
			ix = i + xPos
			iy = j + yPos
			ix1 = ix + int(mag*Hx+0.5)
			iy1 = iy + int(mag*Hy+0.5)
			# line color
			red = 0
			green = 200
			blue = 0
			if (Hy>=0):
				red = 200
			else:
				blue = 200
			if (ix != ix1 or iy != iy1):
				# line(surface, color, start_pos, end_pos, width) -> Rect
				pygame.draw.line(g_screen,(red,green,blue),[ix, iy],[ix1, iy1], 1)	

def drawScreen(drawMode):
	# clear screen
	g_screen.fill(g_backgroundColor)
	
	# draw box
	# draw rect(surface, color, rect) -> Rect (filled box)
	# draw rect outline: rect(surface, color, rect, width, ... ) -> Rect
	pygame.draw.rect(g_screen, g_boxColor, g_boxRect)

	# draw caption
	yOffset = g_yBoxPos + g_boxHeight + 10
	drawText('key menu  [0]:free space, [1]:glass(n=2.0), [2]:metal, [3]:absorber', 20, 3)
	drawText('field= 240x240     [esc]:quit, [p]:pause/go, [d]:draw mode', 20, 23)
	drawText('time = '+str(round(g_sysTime,2)),20,yOffset)
	drawText('theme %d  material: %s ' % (g_theme, g_themeStr[g_theme]), 20, yOffset+20)
	drawText('draw mode: %s wave' % ('Ez','Hxy')[g_drawMode], 250, yOffset+20)
	
	if (drawMode==0):
		#drawEzWave()
		drawEzWaveRes1()
	else:
		drawHxHyWave(20.0)
	
	# Update the full display Surface to the screen: flip() -> None
	pygame.display.flip()


#-------------  main  --------------------------------------------------------------

# set potential and wave function  setInitialCondition(theme)
setInitialCondition(1)

while True:
	if g_pauseFlag==0:
		# evolveField(nCalc,nWaves)
		evolveField(1,g_nWaves)
	
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
	
	if g_inkey == K_p: g_pauseFlag = (g_pauseFlag+1)%2
	elif g_inkey == K_0: setInitialCondition(0) # setInitialCondition(theme)
	elif g_inkey == K_1: setInitialCondition(1)
	elif g_inkey == K_2: setInitialCondition(2)
	elif g_inkey == K_3: setInitialCondition(3)
	elif g_inkey == K_d: g_drawMode = (g_drawMode+1)%2
	g_inkey = 0

	#pygame.time.delay(5)

# end

