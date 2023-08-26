#-----------------------------------------------------------------------------------
#
#  py001_ArMD2D
#    Copyright(C) 2021 Mitsuru Ikeuchi
#    Released under the MIT license ( https://opensource.org/licenses/MIT )
#
#    ver 0.0.0  2021.11.16 created, last updated on 2021.11.23
#
#--------------------  molecular dynamics 2D ----------------------------------------
#
#    time evolution: velocity Verlet Algorithm
#      (1) vi = vi + (Fi/mi)*(0.5dt)
#      (2) ri = ri + vi*dt
#      (3) calculation Fi <- {r1,r2,...,rn} Fi=sum(Fij,j=1 to n,j!=i), Fij=F(ri-rj)
#      (4) vi = vi + (Fi/mi)*(0.5dt)
#      goto (1)
#
#    potential: Lennard-Jones V(r) = 4.0*epsilon*((sigma/r)^12-(sigma/r)^6)
#    force: F(r) = -dV(r)/dr = 24.0*epsilon*r6*(2.0*r6-1.0)/r, r6=(sigma/r)^6
#
#-----------------------------------------------------------------------------------

import sys, pygame
from pygame.locals import *
import math, random


pygame.init()
pygame.display.set_caption("py001 Ar molecular dynamics 2D")

#-------------  set global  --------------------------------------------------------

gc_AMU = 1.66053904e-27             # (kg) atomic mass unit
gc_kB = 1.380649e-23                # (J/K) Boltzmann's constant
gc_nMax = 100                       # max list length of particles

g_N = 48                            # number of particles
g_sysTime = 0.0                     # (s) system time
g_timeStep = 10.0e-15               # (s) time step dt
g_xMax = 8.0E-9                     # (m) x-size of real box
g_yMax = 8.0E-9                     # (m) y-size of real box
g_kineticEnergy = 0.0               # (J) total kinetic energy
g_potentialEnergy = 0.0             # (J) total potential energy

# lists of particles
g_px = [ 0.0 for i in range(gc_nMax) ]   # x-component of position of particle i
g_py = [ 0.0 for i in range(gc_nMax) ]   # y-component of position of particle i
g_vx = [ 0.0 for i in range(gc_nMax) ]   # x-component of velocity of particle i
g_vy = [ 0.0 for i in range(gc_nMax) ]   # y-component of velocity of particle i
g_fx = [ 0.0 for i in range(gc_nMax) ]   # x-component of force of particle i
g_fy = [ 0.0 for i in range(gc_nMax) ]   # y-component of force of particle i

# #check 'i is global? in [ 0.0 for i in range(gc_nMax) ]' 
# a = [ 0.0 for i in range(10) ]
# print('i=',i) 
# -->  NameError: name 'i' is not defined 

# material data:  Lennard-Jones V(r) = 4.0*epsilon*((sigma/r)^12-(sigma/r)^6)
g_mass = 39.95*gc_AMU               # (kg) mass of Ar
g_sigma = 3.418e-10                 # (m) Lennard-Jones potential sigma for Ar 
g_epsilon = 1.711e-21               # (J) Lennard-Jones potential epsilon for Ar

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

g_particleColor= (125,250,0)
g_velocityColor = (200,100,100)

g_inkey = 0
g_pauseFlag = 0 # pauseFlag 0:evolve, 1:pause


#-------------  define function  ---------------------------------------------------

def systemTemperature():
	ek = 0.0
	for i in range(g_N):
		ek += 0.5*g_mass*(g_vx[i]*g_vx[i]+g_vy[i]*g_vy[i])
	return ek/(g_N*gc_kB) # for 2D
	
def ajustVelocity(temp):
	a = math.sqrt(temp/systemTemperature())
	for i in range(g_N):
		g_vx[i] = a*g_vx[i]
		g_vy[i] = a*g_vy[i]


#------------  set initial condition  ----------------------------------------------

# normal distributed random number: -3.0 <= normalRandom3() < 3.0
def normalRandom3():
	return random.random()+random.random()+random.random()+random.random()+random.random()+random.random()-3.0

def setParticlePosition(nn):
	nRow = math.ceil(math.sqrt(nn-0.5))
	ax = g_xMax/(nRow+1)
	ay = g_yMax/(nRow+1)
	for i in range(nn):
		nx = i%nRow
		ny = int(i/nRow)
		g_px[i] = ax*(nx+1)
		g_py[i] = ay*(ny+1)

def setInitialCondition(temp):
	global g_inkey, g_pauseFlag, g_sysTime
	
	g_inkey = 0
	g_pauseFlag = 0
	g_sysTime = 0.0
	
	setParticlePosition(g_N)
	
	# set particle velocity and force
	for i in range(g_N):
		g_vx[i] = 600.0*normalRandom3()
		g_vy[i] = 600.0*normalRandom3()
		g_fx[i] = 0.0
		g_fy[i] = 0.0

	ajustVelocity(temp)


#-------------  time evolution  ----------------------------------------------------

# boundary:L-J type; epsilon = 0.5*epsilonOfAr, sigma = sigmaOfAr
def boundaryForce(r):
	global g_potentialEnergy
	ri = (g_sigma/r)
	r6 = ri*ri*ri*ri*ri*ri
	g_potentialEnergy += 4.0*0.5*g_epsilon*r6*(r6-1.0)
	return (24.0*0.5*g_epsilon*r6*(2.0*r6-1.0)/r)

def calcForce():
	global g_potentialEnergy
	
	g_potentialEnergy = 0.0
	s05 = 0.5*g_sigma
	for i in range(g_N):
		g_fx[i] = 0.0
		g_fy[i] = 0.0
	i = 0
	while i < g_N:
		j = i+1
		while j < g_N:
			xij=g_px[i]-g_px[j]
			yij=g_py[i]-g_py[j]
			r = math.sqrt(xij*xij+yij*yij)
			# calc. L-J force and potential
			ri = (g_sigma/r)
			r6 =ri*ri*ri*ri*ri*ri
			# V(r) = 4.0*epsilon*((sigma/r)^12-(sigma/r)^6)
			g_potentialEnergy += 4.0*g_epsilon*r6*(r6-1.0)
			# F(r) = 24.0*epsilon*r6*(2.0*r6-1.0)/r, r6=(sigma/r)^6
			f = 24.0*g_epsilon*r6*(2.0*r6-1.0)/r
			
			fxij = f*xij/r
			fyij = f*yij/r
			g_fx[i] += fxij
			g_fy[i] += fyij
			g_fx[j] -= fxij
			g_fy[j] -= fyij
			j += 1
		i += 1
	for i in range(g_N):
		g_fx[i] += boundaryForce(g_px[i]+s05) + boundaryForce(g_px[i]-g_xMax-s05)
		g_fy[i] += boundaryForce(g_py[i]+s05) + boundaryForce(g_py[i]-g_yMax-s05)

# velocity Verlet
def moveParticles(dt):
	global g_kineticEnergy
	
	a = 0.5*dt/g_mass
	for i in range(g_N):
		g_vx[i] += a*g_fx[i]
		g_vy[i] += a*g_fy[i]
		g_px[i] += g_vx[i]*dt
		g_py[i] += g_vy[i]*dt
	calcForce()
	for i in range(g_N):
		g_vx[i] += a*g_fx[i]
		g_vy[i] += a*g_fy[i]
	g_kineticEnergy = 0.0
	for i in range(g_N):
		g_kineticEnergy += 0.5*g_mass*(g_vx[i]*g_vx[i]+g_vy[i]*g_vy[i])

def timeEvolution(nCalc,dt):
	global g_sysTime
	
	for i in range(nCalc):
		g_sysTime += dt
		moveParticles(dt)


#-------------  draw  --------------------------------------------------------------

def drawText(txt,x,y):
	# render(text, antialias, color, background=None) -> Surface
	text = g_font.render(txt, 1, g_textColor)
	# blit(source, dest, area=None, special_flags=0) -> Rect
	g_screen.blit(text, (x,y))

def drawParticles():
	sc = g_boxWidth/g_xMax
	xPos = g_xBoxPos
	yPos = g_yBoxPos
	ir = int(0.5*g_sigma*sc+0.5)
	for i in range(g_N):
		ix = int(g_px[i]*sc+0.5)+xPos
		iy = int(g_py[i]*sc+0.5)+yPos
		# circle(surface, color, center, radius) -> Rect
		pygame.draw.circle(g_screen, g_particleColor, [ix, iy], ir)

def drawParticleVelocity(vMag):
	#vMag: velocity line length = v*dt*vMag
	sc = g_boxWidth/g_xMax
	xPos = g_xBoxPos
	yPos = g_yBoxPos
	for i in range(g_N):
		ix = int(g_px[i]*sc+0.5)+xPos
		iy = int(g_py[i]*sc+0.5)+yPos
		ix2 = ix + int(g_vx[i]*g_timeStep*sc*vMag)
		iy2 = iy + int(g_vy[i]*g_timeStep*sc*vMag)
		# line(surface, color, start_pos, end_pos, width) -> Rect
		pygame.draw.line(g_screen,g_velocityColor,[ix, iy],[ix2, iy2], 1)

def drawScreen():
	# clear screen
	g_screen.fill(g_backgroundColor)
	
	# draw box
	# rect(surface, color, rect) -> Rect
	# draw rect outline: rect(surface, color, rect, width, ... ) -> Rect
	pygame.draw.rect(g_screen, g_boxColor, g_boxRect)
	
	# draw caption
	yOffset = g_yBoxPos + g_boxHeight + 10
	drawText('box = %s x %s (nm),   Ar,   N = %d,   [p] pause/go,   [esc] quit' % \
				( str(round(g_xMax*1.0e9,1)), str(round(g_yMax*1.0e9,1)), g_N ), 20, 3)
	drawText('time(ps) = '+str(round(g_sysTime*1.0e12,1)),20,yOffset)
	drawText('Temp(K) = '+str(round(systemTemperature(),1)), 220, yOffset)
	drawText('Total Energy(J) = '+str(round((g_kineticEnergy+g_potentialEnergy)*1.0e19,4))+'E-19', 20, yOffset+20)
	drawText('inter atomic potential: Lennard-Jones', 20, yOffset+40)
	
	# draw particles
	drawParticles()
	
	# draw velocity line
	drawParticleVelocity(200.0) #drawParticleVelocity(vMag) 
	
	# Update the full display Surface to the screen: flip() -> None
	pygame.display.flip()


#-------------  main  --------------------------------------------------------------

# set particles  setInitialCondition(temp)
setInitialCondition(300.0)

while True:
	if g_pauseFlag==0:
		# timeEvolution(nCalc, dt)
		timeEvolution(20,g_timeStep)
	
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
	
	if g_inkey == K_p:
		g_pauseFlag = (g_pauseFlag+1)%2
		g_inkey = 0
	#pygame.time.delay(2)

# end

