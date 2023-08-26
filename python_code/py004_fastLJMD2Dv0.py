#-----------------------------------------------------------------------------------
#
#  py004_fastLJMD2D
#    Copyright(C) 2021 Mitsuru Ikeuchi
#    Released under the MIT license ( https://opensource.org/licenses/MIT )
#
#    ver 0.0.0  2021.11.20 created, last updated on 2021.11.23
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
#    for fast calculation
#      ignore F(r) r>rCutoff 
#      force F(r) <- force table + linear interpolation (see setForceTable() and cutoff(r))
#      register near particles reg[][] (see registerNearParticles()), near means r<rCutoff+20*2000*dt
#        reg[][] use 20 times, assuming particle max speed < 2000m/s
#      force calculation: sum up force(r) (r<rCutoff)
#
#
#-----------------------------------------------------------------------------------

import sys, pygame
from pygame.locals import *
import math, random


pygame.init()
pygame.display.set_caption("py004 fast molecular dynamics LJ-MD2D")

#-------------  set global  --------------------------------------------------------

gc_pi = math.pi                     # math.pi ~ 3.141592653589793
gc_AMU = 1.66053904e-27             # (kg) atomic mass unit
gc_kB = 1.380649e-23                # (J/K) Boltzmann's constant
gc_nMax = 500                       # max list length of particles

g_kind = 2                          # kind of particles
g_N = 100                           # number of particles
g_sysTime = 0.0                     # (s) system time
g_timeStep = 10.0e-15               # (s) time step dt
g_xMax = 8.0E-9                     # (m) x-size of real box
g_yMax = 8.0E-9                     # (m) y-size of real box
g_kineticEnergy = 0.0               # (J) total kinetic energy
g_potentialEnergy = 0.0             # (J) total potential energy
g_rCutoff = 1.0e-9                  # (m) force cutoff length
g_hh = 1.0e-12                      # (m) forceTable r-division r = ir*g_hh

# lists of particles
g_px = [ 0.0 for i in range(gc_nMax) ]   # x-component of position of particle i
g_py = [ 0.0 for i in range(gc_nMax) ]   # y-component of position of particle i
g_vx = [ 0.0 for i in range(gc_nMax) ]   # x-component of velocity of particle i
g_vy = [ 0.0 for i in range(gc_nMax) ]   # y-component of velocity of particle i
g_fx = [ 0.0 for i in range(gc_nMax) ]   # x-component of force of particle i
g_fy = [ 0.0 for i in range(gc_nMax) ]   # y-component of force of particle i

# potential potentialTable[V[0], V[hh], V[2hh],..., V[rCutoff/hh] ] (J)
g_potentialTable = [ 0.0 for i in range(1010) ]
# force table [F[0], F[hh], F[2hh],..., F[rCutoff/hh] ] (N)
g_forceTable = [ 0.0 for i in range(1010) ]
# register i-near particles(j)  (j>i)                         reg[i][j] register particles near i-th particle (j>i)
g_reg = [ [0 for j in range(100)] for i in range(gc_nMax) ] # reg[i][0] total number of particles near i-th particle

# material data:  Lennard-Jones V(r) = 4.0*epsilon*((sigma/r)^12-(sigma/r)^6)
g_mass = 39.95*gc_AMU               # (kg) mass of Ar
g_sigma = 3.418e-10                 # (m) Lennard-Jones potential sigma for Ar 
g_epsilon = 1.711e-21               # (J) Lennard-Jones potential epsilon for Ar
                                    # V(r) = 4.0*epsilon*((sigma/r)^12-(sigma/r)^6)

#       mass(kg)        E(J)       sigma(m)   string  color (R,G,B)
g_materialDataTable = [ \
	[   4.003*gc_AMU,  10.2*gc_kB, 2.576e-10, 'He',  (200,  0,200) ], \
	[  20.183*gc_AMU,  36.2*gc_kB, 2.976e-10, 'Ne',  (  0,  0,250) ], \
	[  39.948*gc_AMU, 124.0*gc_kB, 3.418e-10, 'Ar',  (  0,150,250) ], \
	[  83.500*gc_AMU, 190.0*gc_kB, 3.610e-10, 'Kr',  (  0,220,220) ], \
	[ 131.300*gc_AMU, 229.0*gc_kB, 4.055e-10, 'Xe',  (  0,250,150) ], \
	[ 200.590*gc_AMU, 851.0*gc_kB, 2.898e-10, 'Hg',  (  0,250,  0) ], \
	[   2.016*gc_AMU,  33.3*gc_kB, 2.968e-10, 'H2',  (250,  0,  0) ], \
	[  28.013*gc_AMU,  91.5*gc_kB, 3.681e-10, 'N2',  (250, 80,  0) ], \
	[  31.999*gc_AMU, 113.0*gc_kB, 3.433e-10, 'O2',  (250,160,  0) ], \
	[  18.015*gc_AMU, 809.1*gc_kB, 2.641e-10, 'H2O', (250,250,  0) ], \
	[  16.043*gc_AMU, 137.0*gc_kB, 3.822e-10, 'CH2', (160,250,  0) ], \
	[  44.010*gc_AMU, 190.0*gc_kB, 3.996e-10, 'CO2', ( 80,250,  0) ], \
	[  28.011*gc_AMU, 110.0*gc_kB, 3.590e-10, 'CO',  (  0,250, 80) ]] \

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
#g_font = pygame.font.Font(None, 20) # font size 20
g_font = pygame.font.SysFont(None, 20)
g_textColor = (125, 125, 63)

g_particleColor= (125,250,0)
g_velocityColor = (200,100,100)

g_inkey = 0                         # like inkey
g_pauseFlag = 0                     # pauseFlag 0:evolve, 1:pause
g_theme = 2                         # theme 1:Ne 2:Ar 3:Kr 4:Xe 5:Hg
g_themeStr = ('He','Ne','Ar','Kr','Xe','Hg')
g_tempMode = 0                      # tempMode  0:adiabatic  1:constantTemp
g_contTemp = 300.0                  # controled temperture
g_drawMode = 1                      # drawMode 0:particle 1:particle+velocity


#-------------  define function  ---------------------------------------------------

#-------------  utility

def systemTemperature():
	ek = 0.0
	for i in range(g_N):
		ek += 0.5*g_mass*(g_vx[i]*g_vx[i]+g_vy[i]*g_vy[i])
	return ek/(g_N*gc_kB); # for 2D
	
def ajustVelocity(temp):
	a = math.sqrt(temp/systemTemperature())
	for i in range(g_N):
		g_vx[i] = a*g_vx[i];
		g_vy[i] = a*g_vy[i];


#-------------  set potentialTable[ir] and forceTable[ir]

def cutoff(r):
	ret =0.0
	if (r>0 and r<0.8*g_rCutoff):
		ret = 1.0
	elif ( r>=0.8*g_rCutoff and r<g_rCutoff ):
		ret = 0.5+0.5*math.cos(gc_pi*(r-0.8*g_rCutoff)/(0.2*g_rCutoff))
	return ret

def setPotentialAndForceTable():
	for ir in range(1,1002):
		r = ir*g_hh
		ri = (g_sigma/r)
		r6 = ri*ri*ri*ri*ri*ri
		# V(r) = 4*epsilon*((sigma/r)^12-(sigma/r)^6)
		g_potentialTable[ir] = cutoff(r)*4.0*g_epsilon*r6*(r6-1.0)
		#forceTable[ir] = cutoff(r)*(24.0*epsilon*r6*(2.0*r6-1.0)/r)
	g_potentialTable[0] = g_potentialTable[1] + g_potentialTable[2]

	for ir in range(1,1001):
		# force(r) = -grad V(r)
		g_forceTable[ir] = -(g_potentialTable[ir+1] - g_potentialTable[ir-1])/(2.0*g_hh)
	g_forceTable[1001] = -(0.0 - g_potentialTable[1000])/(2.0*g_hh)
	g_forceTable[0] = g_forceTable[1]


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

def setInitialCondition(kind,nn,BoxSizeInNM,contTemp):
	# kind: particle kind  1:Ne  2:Ar  3:Kr  4:Xe  5:Hg
	# nn: number of particles
	# boxSizeInNM: boxSize in nm
	# contTemp: system tempeature controled to conTemp(K) 
	global g_theme, g_inkey, g_pauseFlag, g_kind, g_N, g_sysTime, g_xMax, g_yMax
	global g_mass, g_epsilon, g_sigma
	global g_tempMode, g_contTemp, g_drawMode

	g_theme = kind
	g_inkey = 0
	g_pauseFlag = 0
	g_tempMode = 0
	g_contTemp = contTemp
	g_drawMode = 1

	g_kind = kind
	g_N = nn
	g_sysTime = 0.0
	g_xMax = BoxSizeInNM*1.0e-9
	g_yMax = BoxSizeInNM*1.0e-9
	g_mass = g_materialDataTable[g_kind][0]
	g_epsilon = g_materialDataTable[g_kind][1]
	g_sigma = g_materialDataTable[g_kind][2]
	
	setPotentialAndForceTable()
	
	setParticlePosition(g_N)
	
	# set particle velocity and force
	for i in range(g_N):
		g_vx[i] = 600.0*normalRandom3()
		g_vy[i] = 600.0*normalRandom3()
		g_fx[i] = 0.0
		g_fy[i] = 0.0
	ajustVelocity(contTemp)


#-------------  time evolution  ----------------------------------------------------

#----- registeration

def registerNearParticles():
	rCut = g_rCutoff+20*2000*g_timeStep
	rcut2 = rCut*rCut
	for i in range(g_N-1):
		k = 1
		for j in range(i+1,g_N):
			r2 = (g_px[i]-g_px[j])*(g_px[i]-g_px[j])+(g_py[i]-g_py[j])*(g_py[i]-g_py[j])
			if (r2<rcut2):
				g_reg[i][k] = j
				k = k + 1;
		g_reg[i][0] = k;

def maxNearParticles():
	mx = 0
	for i in range(g_N-1):
		if (mx<g_reg[i][0]): mx = g_reg[i][0]
	return (mx-1)

#----- force

def force(r):
	# forceTable - linear interporation
	global g_potentialEnergy

	ir = math.floor(r/g_hh)
	a = r - ir*g_hh
	g_potentialEnergy += ((g_hh-a)*g_potentialTable[ir] + a*g_potentialTable[ir+1])/g_hh
	return ((g_hh-a)*g_forceTable[ir] + a*g_forceTable[ir+1])/g_hh

def boundaryForce(r):
	# L-J force
	global g_potentialEnergy

	ri = (g_sigma/r)
	r6 = ri*ri*ri*ri*ri*ri
	g_potentialEnergy += 4.0*0.5*g_epsilon*r6*(r6-1.0)
	return (24.0*0.5*g_epsilon*r6*(2.0*r6-1.0)/r)

def calcForce():
	global g_potentialEnergy

	s05 = 0.5*g_sigma;
	g_potentialEnergy = 0.0;
	for i in range(g_N):
		g_fx[i]=0
		g_fy[i]=0
	
	for i in range(g_N-1):
		for k in range(1,g_reg[i][0]):
			j = g_reg[i][k]
			xij=g_px[i]-g_px[j]
			yij=g_py[i]-g_py[j]
			rij = math.sqrt(xij*xij+yij*yij)
			if (rij<g_rCutoff):
				f = force(rij)
				fxij = f*xij/rij
				fyij = f*yij/rij
				g_fx[i] += fxij
				g_fy[i] += fyij
				g_fx[j] -= fxij
				g_fy[j] -= fyij
	for i in range(g_N):
		g_fx[i] += boundaryForce(g_px[i]+s05)+boundaryForce(g_px[i]-g_xMax-s05)
		g_fy[i] += boundaryForce(g_py[i]+s05)+boundaryForce(g_py[i]-g_yMax-s05)

#----- move particles

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

#----- time evolution

def timeEvolution(tempMode,contTemp):
	# tempMode:  0:adiabatic, 1:controled
	# contTemp:  system tempeature controled to conTemp(K)
	global g_sysTime
	
	if (tempMode==1): ajustVelocity(contTemp)
	
	registerNearParticles()
	for i in range(20):
		g_sysTime += g_timeStep
		moveParticles(g_timeStep)


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

def drawScreen(drawMode):
	# clear screen
	g_screen.fill(g_backgroundColor)
	
	# draw box
	# rect(surface, color, rect) -> Rect
	# draw rect outline: rect(surface, color, rect, width, ... ) -> Rect
	pygame.draw.rect(g_screen, g_boxColor, g_boxRect)

	# draw caption
	yOffset = g_yBoxPos + g_boxHeight + 10
	drawText('key menu    [1]:Ne, [2]:Ar, [3]:Kr, [4]:Xe, [5]:Hg,  [d]:disp mode', 20, 3)
	drawText('[t]:temp mode, controled temp [j]:Temp-10, [k]:Temp+10', 20, 23)
	drawText('box = %s x %s (nm),  %s  N = %d,  [esc]:quit, [p]:pause/go' % \
			( str(round(g_xMax*1.0e9,2)), str(round(g_yMax*1.0e9,2)), g_themeStr[g_kind],g_N ), 20, 43)
	
	drawText('time(ps) = '+str(round(g_sysTime*1.0e12,1)),20,yOffset)
	drawText('Temp(K) = %5.1f' % (systemTemperature()), 220, yOffset)
	drawText('Total Energy(J) = '+str(round((g_kineticEnergy+g_potentialEnergy)*1.0e19,4))+'E-19', 20, yOffset+20)
	if (g_tempMode==1):
		drawText('controled Temp(K) = %5.1f' % g_contTemp, 220, yOffset+20)
	elif (g_tempMode==0):
		drawText('adiabatic (contTemp(K) = %5.1f)' % g_contTemp, 220, yOffset+20)
	drawText('inter atomic potential: Lennard-Jones     maxReg=%d' % (maxNearParticles()), 20, yOffset+40)
	
	if (drawMode==0):
		drawParticles()
	elif (drawMode==1):
		drawParticles()
		drawParticleVelocity(200.0)
	
	# Update the full display Surface to the screen: flip() -> None
	pygame.display.flip()


#-------------  main  --------------------------------------------------------------

# setInitialCondition(kind,nn,BoxSizeInNM,contTemp)
setInitialCondition(2,100, 8.0, 300.0)

while True:
	if g_pauseFlag==0:
		timeEvolution(g_tempMode,g_contTemp)
	
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
		g_pauseFlag = (g_pauseFlag+1)%2
	elif g_inkey == K_1:
		g_inkey = 0
		setInitialCondition(1,100, 8.0, 300.0)
	elif g_inkey == K_2:
		g_inkey = 0
		setInitialCondition(2,100, 8.0, 300.0)
	elif g_inkey == K_3:
		g_inkey = 0
		setInitialCondition(3,100, 8.0, 300.0)
	elif g_inkey == K_4:
		g_inkey = 0
		setInitialCondition(4,100, 8.0, 300.0)
	elif g_inkey == K_5:
		g_inkey = 0
		setInitialCondition(5,100, 8.0, 300.0)
	elif g_inkey == K_d:
		g_inkey = 0
		g_drawMode = (g_drawMode+1)%2
	elif g_inkey == K_t:
		g_inkey = 0
		g_tempMode = (g_tempMode+1)%2
	elif g_inkey == K_j:
		g_inkey = 0
		g_contTemp -= 10.0
		if (g_contTemp<10.0):g_contTemp = 10.0 
	elif g_inkey == K_k:
		g_inkey = 0
		g_contTemp += 10.0
		if (g_contTemp>600.0):g_contTemp = 600.0 
	
	pygame.time.delay(5)

# end

