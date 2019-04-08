import numpy as np
import math
import multiprocessing
import time
import sys

import MG1DMC_01_Utilities


VACUUM = 0
ISOTROPIC = 1
REFLECTIVE = 2
LEFT = 0
RIGHT = 1

X = 0
Y = 1
Z = 2



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
class RandomNumberGenerator:
  def __init__(self):
    self.RNGen = np.random.RandomState()
    self.total_calls = 0
    self.cumulative = 0.0

  def RN(self):
    self.total_calls += 1
    numb = self.RNGen.uniform(0.0,1.0)
    self.cumulative += numb

    return numb

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
class Tally:
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Constructor
  # batch_size: The amount of contributions to
  # assimilate before making an estimation of the
  # mean.
  def __init__(self,batch_size=1000):
    self.batch_size = batch_size

    self.v_sx = 0.0                    # Accumulator of tally value
    self.v_sx_contrib_count = 0        # Counts contributions from unique indexes
    self.last_index = -1

    self.mu = 0.0                      # Running average
    self.mu_sx = 0.0                   # Accumulator of tally mean value
    self.mu_sx2 = 0.0                  # Accumulator of tally mean squared value
    self.mu_sx_contrib_count=0         # Counts contributions to mean value


  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Contribute
  def Contrib(self,value,contrib_index=-1):
    self.v_sx    += value

    if contrib_index!=self.last_index:
      self.v_sx_contrib_count = contrib_index
      # new_mean = self.v_sx/self.v_sx_contrib_count
      # self.mu = new_mean
      # self.mu_sx += new_mean
      # self.mu_sx2 += new_mean*new_mean
      # self.mu_sx_contrib_count+=1

  def StdDev(self):
    binv = 1.0/self.mu_sx_contrib_count

    Sx = self.mu_sx
    Sx2= self.mu_sx2

    return math.sqrt(binv*(Sx2 - binv*Sx*Sx))

  def Mean(self):
    binv = 1.0/self.v_sx_contrib_count

    return binv*self.v_sx






# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
class MultiGrp1DMC(MG1DMC_01_Utilities.MG1DMC_Methods):
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Constructor
  def __init__(self,L,G,mesh,materials,source,bcs,group_struct,num_threads=1):
    self.L=L
    self.G=G
    self.mesh=mesh
    self.materials=materials
    self.source=source
    self.bcs=bcs
    self.group_struct = group_struct

    self.epsilon = 1.0e-8

    self.num_threads = num_threads

    self.avg_mu = 0.0
    self.cumul_mu = 0.0
    self.mu_calls = 0

    self.rngen = RandomNumberGenerator()

    self.SourceRoutine = self.GetSourceParticle

    self.RMC_active = False
    self.RMC_mesh = []
    self.RMC_cur_elem = []
    self.RMC_element_CDF = []
    self.RMC_jumps = []
    self.RMC_avg_flux = []
    self.RMC_res_tot = 0.0
    self.RMC_res_tot_set = False



    # Default tallies for each process
    # These are sums
    self.glob_tally = np.zeros(self.G)
    self.elem_tally = np.zeros((self.G,self.mesh.Ndiv))

    self.elem_tally2=[]
    for g in range(0,self.G):
      mesh_tally=[]
      for k in range(0, self.mesh.Ndiv):
        mesh_tally.append(Tally())
      self.elem_tally2.append(mesh_tally)



    # Combined tallies across processes
    self.comb_glob_tally=np.zeros(self.G)
    self.comb_elem_tally=np.zeros((self.G, self.mesh.Ndiv))

    self.comb_elem_tally2=[]
    for g in range(0, self.G):
      mesh_tally=[]
      for k in range(0, self.mesh.Ndiv):
        mesh_tally.append(Tally())
      self.comb_elem_tally2.append(mesh_tally)



    self.mesh.elements[0].xi__ -= self.epsilon
    self.mesh.elements[self.mesh.Ndiv-1].xip1 +=self.epsilon

    self.outputFileName=""
    self.outputFileNameSet=False

    self.particles_ran=0
    self.processQ = []     #Process que to share tallies
    self.threads=[]

    self.num_batches = 0


  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Combine tallies
  def CombineTallies(self,process_que):
    for thd in range(0,self.num_threads):
      que = process_que[thd].get()

      que_glob_tally=que[0]
      for g in range(0, self.G):
        self.comb_glob_tally[g] += que_glob_tally[g]

      que_elem_tally=que[1]
      for g in range(0, self.G):
        for k in range(0,self.mesh.Ndiv):
          self.comb_elem_tally[g,k] += que_elem_tally[g,k]

      que_elem_tally2=que[2]
      for g in range(0, self.G):
        for k in range(0,self.mesh.Ndiv):
          self.comb_elem_tally2[g][k].v_sx += \
            que_elem_tally2[g][k].v_sx
          self.comb_elem_tally2[g][k].v_sx_contrib_count+= \
            que_elem_tally2[g][k].v_sx_contrib_count

          # self.comb_elem_tally2[g][k].mu+= \
          #   que_elem_tally2[g][k].mu/self.num_threads/self.num_batches
          # self.comb_elem_tally2[g][k].mu_sx+= \
          #   que_elem_tally2[g][k].mu_sx
          # self.comb_elem_tally2[g][k].mu_sx2+= \
          #   que_elem_tally2[g][k].mu_sx2
          # self.comb_elem_tally2[g][k].mu_sx_contrib_count+= \
          #   que_elem_tally2[g][k].mu_sx_contrib_count
          mean = que_elem_tally2[g][k].v_sx / \
                 self.comb_elem_tally2[g][k].v_sx_contrib_count
          self.comb_elem_tally2[g][k].mu_sx += mean
          self.comb_elem_tally2[g][k].mu_sx2+= mean*mean
          self.comb_elem_tally2[g][k].mu_sx_contrib_count+=1


  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Run the forward problem
  def RunMTForward(self,num_particles,batch_size,report_intvl=10000,verbose=False):
    print("Running MC solver (forward)")
    particles_per_thread = int(batch_size/self.num_threads)
    self.num_batches = int(num_particles/batch_size)

    t_cycle=time.time()
    print("Number of particles total %d"%num_particles)
    print("Batch size %d"%batch_size)
    print("Number of particles per thread %d"%particles_per_thread)
    for b in range(0,self.num_batches):

      t_cycle = time.time()

      self.threads=[]
      self.processQ = []
      for th in range(0, self.num_threads):
        self.processQ.append(multiprocessing.Queue())
        self.threads.append(multiprocessing.Process(target=self.WorkFunction,
                            args=(particles_per_thread, th, self.processQ[th])))
        self.threads[th].start()


      for th in range(0, self.num_threads):
        self.threads[th].join()
      self.particles_ran += batch_size

      self.CombineTallies(self.processQ)

      t_cycle=time.time()-t_cycle
      perf=batch_size/t_cycle
      print("Particles=%7d, rate=%.0f part/sec "%(self.particles_ran, perf))

      self.WriteRestartData()



  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Thread work function
  def WorkFunction(self,num_particles,thread_id,processQ):
    # =========================================== Initialize RN generator
    t=int(time.time()*1000.0)
    np.random.seed(thread_id + self.particles_ran)
    self.rngen = RandomNumberGenerator()

    # =========================================== Sample particles
    for p in range(0,num_particles):
      particle = self.SourceRoutine(p+1)

      while (particle.alive):
        particle = self.RayTrace(particle,thread_id)

    if self.mu_calls>0:
      self.avg_mu = self.cumul_mu/self.mu_calls

    tallies=[]
    tallies.append(self.glob_tally)
    tallies.append(self.elem_tally)
    tallies.append(self.elem_tally2)
    processQ.put(tallies)

    # print("Average RN: %g" %(self.rngen.cumulative/self.rngen.total_calls))


  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Zero-out the tallies
  def ZeroOutTallies(self):
    self.particles_ran = 0
    self.comb_glob_tally=np.zeros(self.G)
    self.comb_elem_tally=np.zeros((self.G, self.mesh.Ndiv))

    self.comb_elem_tally2=[]
    for g in range(0, self.G):
      mesh_tally=[]
      for k in range(0, self.mesh.Ndiv):
        mesh_tally.append(Tally())
      self.comb_elem_tally2.append(mesh_tally)



