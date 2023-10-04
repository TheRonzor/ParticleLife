import math
import numpy as np

from traceback import print_exc

class Particle:
    def __init__(self, 
                 type:   int,
                 size:   int,
                 color:  np.array,
                 pos:    np.array,
                 vel:    np.array,
                 index:  int
                 ):
        
        self.type  = type
        self.size  = size
        self.color = color
        self.pos   = pos
        self.vel   = vel
        self.index = index

        return

class World:
    particles  = [] # list of Particle(s)
    types      = {} # dictionary of particle types (size and color)

    pos_array: np.array     # positions
    vel_array: np.array     # velocities
    col_array: np.array     # RGBA
    siz_array: np.array     # particle sizes
    typ_array: np.array

    col_unique = [] # possibly unneeded
    siz_unique = [] # possibly unneeded

    WORLD_SIZE      = 1
    VELOCITY_SCALE  = 0.1
    PARTICLE_LOC    = 0.5
    PARTICLE_SCALE  = 0.01

    DT              = 0.01
    FORCE_DIV       = 10

    DEFAULT_R       = [0.05, 0.075, 0.150, 0.20]
    DEFAULT_F       = [-1, 1]
    DEFAULT_B       = 0.9

    ALPHA_MIN       = 0.5
    ALPHA_MAX       = 0.8

    def __init__(self,
                 damping = DEFAULT_B,
                 jitter  = 1e-4,
                 r       = DEFAULT_R.copy(),
                 f       = DEFAULT_F.copy(),
                 p       = 2,
                 prec    = np.float64,
                 seed    = None
                 ):
        self.damping = damping
        self.jitter  = jitter

        self.r       = r
        self.f       = f
        self.p       = p
        self.update_force()

        self.prec = prec
        self.eps = np.finfo(self.prec).eps

        if seed is None:
            seed = np.random.randint(10000)
        self.update_rng(seed)
        return
    
    def update_rng(self, seed):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        #print('Initialized with seed:', seed)
        return
    
    def create_particle_type(self,
                           type:  int,
                           size:  int = None,
                           color: int = None
                           ):
        if type not in self.types:
            if size is None:
                size = self.rng.randint(2,6)**4
            if color is None:
                color = self.rng.random(size=4)
                color[-1] = max(min(color[-1], self.ALPHA_MAX), self.ALPHA_MIN)
            self.types[type] = {'size': size, 
                                'color': color}
        else:
            raise ValueError('Particle type ' + str(type) + ' already exists')
        return type
    
    def create_particle(self,
                       type: int,
                       pos:  np.array = None,
                       vel:  np.array = np.array([0,0]),
                       ):
        if type not in self.types:
            type = self.create_particle_type(type)
        if pos is None:
            pos = self.rng.normal(loc   = self.PARTICLE_LOC, 
                                  scale = self.PARTICLE_SCALE, 
                                  size  = 2)
        p = Particle(type,
                     self.types[type]['size'], 
                     self.types[type]['color'], 
                     pos,
                     vel, 
                     len(self.particles))
        self.particles.append(p)
        return
    
    def create_particles(self,
                        type:    int,
                        num:     int      = 10,     
                        pos_arr: np.array = None,    # Will override num if passed
                        vel_arr: np.array = None
                        ):
        if pos_arr is None:
            pos_arr = self.rng.normal(loc=0.5, scale=0.1, size=(num, 2))
        if vel_arr is None:
            vel_arr = np.zeros(shape=(pos_arr.shape[0],2))
        for pos, vel in zip(pos_arr, vel_arr):
            self.create_particle(type, pos, vel)
        return
    
    def get_particle(self, index:int)->Particle:
        p = self.particles[index]
        p.pos = self.pos_array[index, :]
        p.vel = self.vel_array[index, :]
        return p
    
    def init_data(self, random_interactions=True):
        self.pos_array = np.zeros(shape=(len(self.particles), 2))
        self.vel_array = np.zeros(shape=(len(self.particles), 2))
        self.col_array = np.zeros(shape=(len(self.particles), 4))
        self.siz_array = np.zeros(shape=(len(self.particles)))
        self.typ_array = np.zeros(shape=(len(self.particles)),dtype=np.int_)

        for i,p in enumerate(self.particles):
            self.pos_array[i, :] = p.pos
            self.vel_array[i, :] = p.vel
            self.col_array[i, :] = p.color
            self.siz_array[i] = p.size
            self.typ_array[i] = p.type
        
        self.num_particles = len(self.particles)

        #self.num_combs = math.comb(self.num_particles, 2) # Requires Python > 3.8
        self.num_combs = 0
        for i in range(len(self.particles)):
            for _ in range(i+1, len(self.particles)):
                self.num_combs+=1
        
        self.dir_array = np.zeros([self.num_combs, 2]).astype(self.prec)
        self.dist_array = np.zeros([self.num_combs, 2]).astype(self.prec)

        if random_interactions:
            self.create_random_interactions()
        else:
            print('Are interactions defined?')
        return
    
    def create_random_interactions(self):
        num_types = len(self.types)
        self.Laws = self.rng.uniform(low=-1, high=1, size=[num_types]*2)
        return
    
    def compute_distances(self):
        '''
        Compute the Minkowski distance between
        all particles and store the distance 
        matrix in condensed form.
        '''
        k=0
        for i in range(self.num_particles):
            for j in range(i+1, self.num_particles):
                self.dir_array[k] = (self.pos_array[j] - self.pos_array[i])
                k+=1
        idx = abs(self.dir_array)>self.WORLD_SIZE/2
        self.dir_array[idx] = (self.WORLD_SIZE - abs(self.dir_array[idx]))*(-np.sign(self.dir_array[idx]))
        self.dist_array = np.sum(np.abs(self.dir_array)**self.p, axis=1)**(1/self.p)
        return
    
    def update_force(self):
        try:
            self.m = [
                    -self.f[0]/self.r[0], 
                    self.f[1]/(self.r[1] - self.r[0]),
                    -self.f[1]/(self.r[2] - self.r[1])
                    ]
        except ZeroDivisionError:
            print('Ignoring brief division by zero, watch those sliders!')
            pass
        return
    
    def compute_force(self, d):
        if d<self.r[0]:
            return (self.m[0]*d+self.f[0])/self.FORCE_DIV
        elif d<self.r[1]:
            return (self.m[1]*(d-self.r[0]))/self.FORCE_DIV
        elif d<self.r[2]:
            return (self.m[2]*(d-self.r[2]))/self.FORCE_DIV
        else:
            return 0
    
    # For working with the condensed distance matrix============================================
    # https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist
    def calc_row_idx(self, k, n):
        return int(math.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))
    def elem_in_i_rows(self, i, n):
        return i * (n - 1 - i) + (i*(i + 1))//2
    def calc_col_idx(self, k, i, n):
        return int(n - self.elem_in_i_rows(i + 1, n) + k)
    def condensed_to_square(self, k, n):
        i = self.calc_row_idx(k, n)
        j = self.calc_col_idx(k, i, n)
        return i, j
    # ==========================================================================================
    
    def update_vel(self):
        self.compute_distances()
        idx = np.where(self.dist_array<self.r[2])[0]

        #self.vel_array *= self.damping

        if self.jitter:
            self.vel_array += self.rng.normal(scale = self.WORLD_SIZE*self.jitter, size=[self.num_particles,2])

        for k in idx:
            i,j, = self.condensed_to_square(k, self.num_particles)
            f = self.compute_force(self.dist_array[k])
            self.dist_array[np.where(self.dist_array == 0)] = self.eps
            base_force = f*(self.dir_array[k]/self.dist_array[k])
            if self.dist_array[k]>self.r[0]:
                self.vel_array[i,:] += base_force*self.Laws[self.typ_array[i],self.typ_array[j]]
                self.vel_array[j,:] -= base_force*self.Laws[self.typ_array[j],self.typ_array[i]]
            else:
                self.vel_array[i,:] += base_force
                self.vel_array[j,:] -= base_force
        self.vel_array *= self.damping
        return
    
    def update_pos(self):
        self.update_vel()
        self.pos_array += self.vel_array*self.DT
        self.pos_array %= self.WORLD_SIZE
        return
    
    def shake(self):
        self.vel_array += self.rng.normal(loc=10, scale=5, size=self.vel_array.shape)
        return
    
    def suck(self):
        self.vel_array = (0.5 - self.pos_array)*self.FORCE_DIV
        return