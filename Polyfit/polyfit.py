import numpy as np
import matplotlib.pyplot as plt

class spline:
    def __init__(self, window_size=5, degree=3, intersected_values=3, y=None, x=None):
        if degree >= window_size:
            raise ValueError("Degree must be less than window size")
        if intersected_values >= window_size:
            raise ValueError("Intersected values must be less than window size")
        self.window_size = window_size
        self.degree = degree
        self.offset = window_size - intersected_values
        self.segments = int(len(y)/self.offset)
        self.coeffs = np.zeros((self.segments, degree+1))
        self.bounds = np.zeros((self.segments, 2))
        self.y = y
        self.x = x

    def fit(self):
        for i in range(self.segments):
            lower_bound = i*self.offset
            upper_bound = min(i*self.offset+self.window_size, len(self.y)-1)
            self.coeffs[i] = np.polyfit(self.x[lower_bound:upper_bound], self.y[lower_bound:upper_bound], self.degree)
            self.bounds[i] = [self.x[lower_bound], self.x[upper_bound]]

    def value(self, x, der=0, tanh=True):
        if x <= self.bounds[0][0]:
            return np.polyval(self.coeffs[0], x)
        elif x >= self.bounds[-1][1]:
            return np.polyval(self.coeffs[-1], x)
        
        y = None
        searching_window = max(10, int(self.segments / 33)) # searching on 3% of the segments
        first = max(0, int((x - self.x[0])/ (self.x[-1] - self.x[0]) * self.segments) - int(searching_window/2))

        for i in range(first, first + searching_window):
            if x < self.bounds[i][1]:
                if der == 0:
                    y = np.polyval(self.coeffs[i], x)
                else:
                    y = np.polyval(np.polyder(self.coeffs[i], der), x)
                
                if i != self.segments - 1 and x > self.bounds[i+1][0]:
                    if tanh:
                        ratio = (np.tanh(2*np.pi*((x-self.bounds[i+1][0])/(self.bounds[i][1]-self.bounds[i+1][0])-0.5))+1)/2
                    else:
                        ratio = (x-self.bounds[i+1][0])/(self.bounds[i][1]-self.bounds[i+1][0])

                    if der == 0:
                        y = y*(1-ratio) + np.polyval(self.coeffs[i+1], x)*ratio
                    else:
                        y = y*(1-ratio) + np.polyval(np.polyder(self.coeffs[i+1], der), x)*ratio
                break
            
        if y is None:
            raise ValueError("Searching window too small -> framerate not constant enough")

        return y

if __name__ == "__main__":
    # Load data
    y = np.load("positions.npy")
    x = np.linspace(0, 1, len(y))
    plt.scatter(x, y, label="raw data")

    # Fit spline
    s = spline(window_size=5, degree=3, intersected_values=3, y=y, x=x)
    s.fit()

    x_spline = np.linspace(0, 1, 10000)
    y_spline = [s.value(x) for x in x_spline]
    y_spline_derivative = [s.value(x, der=1) for x in x_spline]
    y_spline_second_derivative = [s.value(x, der=2) for x in x_spline]

    plt.plot(x_spline, y_spline, label="spline")
    plt.plot(x_spline, y_spline_derivative, label="spline derivative")
    plt.plot(x_spline, y_spline_second_derivative, label="spline second derivative")

    plt.show()


