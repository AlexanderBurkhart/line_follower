import numpy as np

class PID(object):
	def __init__(self, Kp, Ki, Kd):
		self.WINDOW_SIZE = 20		

		self.Kp = Kp
		self.Ki = Ki
		self.Kd = Kd

		self.p_error = 0
		self.i_error = 0
		self.d_error = 0

		self.window = np.zeros(self.WINDOW_SIZE)		
		self.i = self.WINDOW_SIZE-1
		self.sum = 0

	def add_i(self, err):
		self.i = (self.i+1) % self.WINDOW_SIZE
		self.sum = self.sum - self.window[self.i] + err
		return self.sum

	def update_error(self, cte, dt):
		self.d_error = (cte-self.p_error) / dt
		self.p_error = cte
		self.i_error = self.add_i(cte*dt)

	def total_error(self):
		return self.Kp*self.p_error + self.Ki*self.i_error + self.Kd*self.d_error

