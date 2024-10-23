from manim import *


class AnimatedSquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        square = Square()  # create a square

        self.play(Create(square))  # show the square on screen
        self.play(square.animate.rotate(PI / 4))  # rotate the square
        self.play(Transform(square, circle))  # transform the square into a circle
        self.play(
            square.animate.set_fill(PINK, opacity=0.5)
        ) 


print(f"Total ships served: {len(stats.total_times)}")
print(f"Customer average time in simulation = {average_time(stats.total_times)}")
print(f"Customer average time waiting = {average_time(stats.waiting_times)}")
print(f"Customer average time processing = {average_time(stats.processing_times)}")
print(f"Customer average time waiting for meteorological conditions = {average_time(stats.mc_waiting_times)}")
print(f"Customer average berth waiting time = {average_time(stats.berth_waiting_times)}")