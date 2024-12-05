import salabim as sim

class Ship(sim.Component):
    def process(self):
        # Define ship behavior
        # Salabim handles the animation based on the component's state
        yield self.hold(sim.Uniform(1, 5).sample())

env = sim.Environment()
for i in range(5):
    Ship()

env.run(till=100)
env.animate()
