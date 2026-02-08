from manimlib import *
import numpy as np
from scipy.spatial import ConvexHull
from scipy.cluster.vq import whiten, kmeans, vq


class SimpleAxes(VGroup):
    """
    Minimal first-quadrant axes with a c2p mapper.
    Draws only positive x/y arrows starting at origin.
    """


    def __init__(
            self,
            x_max=10, y_max=7,
            x_length=8.8, y_length=5.2,
            origin=ORIGIN + 2.5 * DOWN + 4.4 * LEFT,
            stroke_width=2,
            tip=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.x_length = float(x_length)
        self.y_length = float(y_length)
        self.origin = origin
        self.x_unit = self.x_length / self.x_max
        self.y_unit = self.y_length / self.y_max

        # Positive axes only
        x_end = self.origin + np.array([self.x_length, 0, 0])
        y_end = self.origin + np.array([0, self.y_length, 0])
        if tip:
            self.x_axis = Arrow(start=self.origin, end=x_end, buff=0, stroke_width=stroke_width)
            self.y_axis = Arrow(start=self.origin, end=y_end, buff=0, stroke_width=stroke_width)
        else:
            self.x_axis = Line(self.origin, x_end, stroke_width=stroke_width)
            self.y_axis = Line(self.origin, y_end, stroke_width=stroke_width)

        self.add(self.x_axis, self.y_axis)


    def c2p(self, x, y, z=0.0):
        return self.origin + np.array([x * self.x_unit, y * self.y_unit, 0.0])


class WhatAreClusters(Scene):
    def construct(self):
        # Create axes
        axes = SimpleAxes(
            x_max=10, y_max=7,
            x_length=8.8, y_length=5.2,
            origin=ORIGIN + 2.5 * DOWN + 4.4 * LEFT,
            stroke_width=2, tip=True
        )
        x_label = Tex("x_{1}")
        y_label = Tex("x_{2}")
        x_label.next_to(axes.x_axis.get_end(), DOWN, buff=0.28)
        y_label.next_to(axes.y_axis.get_end(), LEFT, buff=0.28)

        a = VGroup(axes, x_label, y_label).scale(1.17)
        self.play(ShowCreation(a))

        # Create 3 distinct clusters of points - REPOSITIONED FOR AXES
        # Cluster 1 (left) - Red
        cluster1 = [
            (1.2, 3.8), (1.8, 2.9), (0.9, 4.1), (1.5, 3.5), (1.7, 2.7),
            (0.8, 4.3), (2.1, 3.2), (1.4, 3.9), (1.0, 3.1), (2.0, 3.6),
            (1.3, 2.8), (1.6, 4.2), (1.1, 3.7), (1.9, 3.0), (1.7, 3.4)
        ]

        # Cluster 2 (center-top) - Green
        cluster2 = [
            (4.8, 5.2), (5.2, 5.8), (4.4, 4.9), (4.9, 5.5), (5.1, 6.1),
            (4.5, 5.0), (5.3, 5.7), (4.7, 5.3), (4.3, 5.6), (5.4, 5.1),
            (5.0, 6.0), (4.6, 4.8), (4.8, 5.4), (5.2, 5.9), (4.9, 5.2)
        ]

        # Cluster 3 (right) - Blue
        cluster3 = [
            (7.8, 2.1), (8.4, 2.7), (7.5, 1.8), (8.1, 2.4), (8.0, 3.0),
            (8.5, 1.9), (7.7, 2.5), (8.2, 2.2), (7.9, 2.8), (8.3, 2.0),
            (8.0, 2.6), (7.6, 2.3), (8.1, 1.7), (8.4, 2.9), (7.8, 2.4)
        ]

        all_points = cluster1 + cluster2 + cluster3
        colors = [RED] * len(cluster1) + [GREEN] * len(cluster2) + [BLUE] * len(cluster3)

        # Create dots using axes coordinates - SCALED BY 0.7
        dots = VGroup(*[
            Dot(point=axes.c2p(x, y), radius=0.12, color=GREY).scale(0.7)
            for x, y in all_points
        ])

        self.play(FadeIn(dots, lag_ratio=0.05), run_time=2)
        self.wait(1)

        # Color the dots by cluster
        color_animations = []
        for i, color in enumerate(colors):
            color_animations.append(dots[i].animate.set_color(color))

        self.play(*color_animations, run_time=1.5)
        self.wait(1)

        # Create ELLIPSES around each cluster
        clusters_data = [
            (cluster1, RED, "Cluster 1"),
            (cluster2, GREEN, "Cluster 2"),
            (cluster3, BLUE, "Cluster 3")
        ]

        shapes = []
        labels = []

        for cluster_points, color, label_text in clusters_data:
            # Convert to axes coordinates
            axes_points = [axes.c2p(x, y) for x, y in cluster_points]
            points_array = np.array([[p[0], p[1]] for p in axes_points])
            centroid = np.mean(points_array, axis=0)

            # Calculate the spread of points to determine ellipse size
            x_spread = np.max(points_array[:, 0]) - np.min(points_array[:, 0])
            y_spread = np.max(points_array[:, 1]) - np.min(points_array[:, 1])

            # Create ellipse with some padding around the points
            ellipse_width = x_spread * 1.4 + 0.5
            ellipse_height = y_spread * 1.4 + 0.5

            # Create the ellipse
            ellipse = Ellipse(
                width=ellipse_width,
                height=ellipse_height,
                fill_color=color,
                fill_opacity=0.25,
                stroke_color=color,
                stroke_width=3
            )
            ellipse.move_to(np.array([centroid[0], centroid[1], 0]))
            shapes.append(ellipse)

            # Add cluster label
            label = Text(label_text, font_size=20, weight=BOLD).set_color(color)
            label.move_to(np.array([centroid[0], centroid[1], 0]))
            label.shift(DOWN * 1.5)
            label.scale(2)
            labels.append(label)

        # Show ellipses and labels
        self.play(*[FadeIn(shape) for shape in shapes], run_time=1.5)
        self.play(*[Write(label) for label in labels], run_time=1)
        self.wait(2)


class KMeansClusteringNonMathematical(VGroup):
    """
    Minimal first-quadrant axes with a c2p mapper.
    Draws only positive x/y arrows starting at origin.
    """


    def __init__(
            self,
            x_max=10, y_max=7,  # expanded extents for more spread
            x_length=8.8, y_length=5.2,  # on-screen lengths
            origin=ORIGIN + 2.5 * DOWN + 4.4 * LEFT,
            stroke_width=2,
            tip=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.x_length = float(x_length)
        self.y_length = float(y_length)
        self.origin = origin
        self.x_unit = self.x_length / self.x_max
        self.y_unit = self.y_length / self.y_max

        # Positive axes only
        x_end = self.origin + np.array([self.x_length, 0, 0])
        y_end = self.origin + np.array([0, self.y_length, 0])
        if tip:
            self.x_axis = Arrow(start=self.origin, end=x_end, buff=0, stroke_width=stroke_width)
            self.y_axis = Arrow(start=self.origin, end=y_end, buff=0, stroke_width=stroke_width)
        else:
            self.x_axis = Line(self.origin, x_end, stroke_width=stroke_width)
            self.y_axis = Line(self.origin, y_end, stroke_width=stroke_width)

        self.add(self.x_axis, self.y_axis)


    def c2p(self, x, y, z=0.0):
        return self.origin + np.array([x * self.x_unit, y * self.y_unit, 0.0])


    def get_x_end(self):
        return self.x_axis.get_end()


    def get_y_end(self):
        return self.y_axis.get_end()


class KMeansIntroduction(Scene):
    def construct(self):

        self.camera.frame.scale(1.08)
        # ------------------- Tunables (sizes kept) -------------------
        DOT_RADIUS = 0.14
        CENTROID_RADIUS = 0.28
        CENTROID_STROKE = 4
        LINE_STROKE = 5  # thicker dashed lines
        DASH_LEN = 0.14
        MAX_ITERS = 10
        DEMO_ITERS = 2  # Only show dotted lines for first 2 iterations
        TOL = 1e-6  # tighter so convergence needs multiple moves

        # Colors
        DOT_RED = RED  # dots use Manim RED/BLUE
        DOT_BLUE = BLUE
        PURE_RED = "#FF0000"  # centroids are pure hex
        PURE_BLUE = "#0000FF"

        # Z-index layering: dots < dashed lines < centroids < centroid labels
        Z_DOTS = 0
        Z_LINES = 1
        Z_CENTROIDS = 3
        Z_CENTROID_LABELS = 4


        # ------------------- Helpers -------------------
        def color_for(assign):
            return DOT_RED if assign == "red" else DOT_BLUE


        def dist(p, q):
            return np.linalg.norm(p - q)


        def make_X():
            try:
                t = Text("X", weight=BOLD)
            except Exception:
                t = Tex("\\mathbf{X}")
            t.set_stroke(width=1)
            return t


        # ------------------- Axes -------------------
        axes = SimpleAxes(
            x_max=10, y_max=7,
            x_length=8.8, y_length=5.2,
            origin=ORIGIN + 2.5 * DOWN + 4.4 * LEFT,
            stroke_width=2, tip=True
        )
        x_label = Tex("x_{1}")
        y_label = Tex("x_{2}")
        x_label.next_to(axes.get_x_end(), DOWN, buff=0.28)
        y_label.next_to(axes.get_y_end(), LEFT, buff=0.28)

        # ------------------- Points: Two VERY dispersed clusters -------------------
        # Cluster A (left region): WIDELY scattered across left half
        cluster_A = [
            (0.8, 1.2), (1.5, 4.8), (0.6, 6.1), (2.2, 0.9), (1.9, 5.7),
            (0.9, 3.4), (2.8, 2.1), (1.1, 6.3), (2.6, 4.2), (0.7, 2.8),
            (1.7, 1.5), (3.1, 5.9), (0.5, 4.1), (2.4, 3.7), (1.3, 0.7),
            (2.9, 6.2), (1.8, 2.3), (0.4, 5.4), (2.7, 1.8), (1.2, 3.9),
            (3.2, 0.8), (0.8, 5.1), (2.1, 4.6), (1.6, 1.1), (2.5, 6.0)
        ]

        # Cluster B (right region): WIDELY scattered across right half
        cluster_B = [
            (6.2, 5.8), (7.8, 1.4), (8.9, 4.2), (6.7, 2.8), (9.1, 5.9),
            (7.4, 0.8), (8.1, 6.1), (6.9, 3.7), (9.3, 2.1), (7.2, 4.9),
            (8.7, 1.7), (6.4, 5.2), (9.0, 3.4), (7.6, 6.4), (8.3, 0.9),
            (6.8, 4.1), (9.2, 2.6), (7.1, 5.7), (8.5, 3.8), (6.5, 1.3),
            (9.4, 4.8), (7.9, 2.4), (8.0, 5.4), (6.6, 3.1), (8.8, 6.2)
        ]

        points_coords = cluster_A + cluster_B
        nA = len(cluster_A)
        A_indices = list(range(0, nA))
        B_indices = list(range(nA, len(points_coords)))

        points = VGroup(*[
            Dot(axes.c2p(x, y), radius=DOT_RADIUS, color=GREY).set_z_index(Z_DOTS)
            for x, y in points_coords
        ])

        a = VGroup(axes, x_label, y_label).scale(1.17)
        self.play(ShowCreation(a))

        self.play(FadeIn(points, lag_ratio=0.02, run_time=1.0))

        # ------------------- Centroids: Start FAR from actual clusters -------------------
        # Red centroid starts at top-middle (far from both clusters)
        red_centroid = Circle(
            radius=CENTROID_RADIUS,
            stroke_color=PURE_RED, fill_color=PURE_RED,
            stroke_width=CENTROID_STROKE, fill_opacity=1.0
        ).move_to(axes.c2p(5.0, 6.5)).set_z_index(Z_CENTROIDS)  # Top-middle
        red_X = make_X().move_to(red_centroid.get_center()).set_z_index(Z_CENTROID_LABELS)

        # Blue centroid starts at bottom-middle (far from both clusters)
        blue_centroid = Circle(
            radius=CENTROID_RADIUS,
            stroke_color=PURE_BLUE, fill_color=PURE_BLUE,
            stroke_width=CENTROID_STROKE, fill_opacity=1.0
        ).move_to(axes.c2p(5.0, 0.5)).set_z_index(Z_CENTROIDS)  # Bottom-middle
        blue_X = make_X().move_to(blue_centroid.get_center()).set_z_index(Z_CENTROID_LABELS)

        self.play(
            ShowCreation(red_centroid), Write(red_X),
            ShowCreation(blue_centroid), Write(blue_X)
        )

        # Pointers to rotate demo choices each iteration
        a_ptr, b_ptr = 0, 0
        alternate_order = True  # alternate demo order each iteration to vary RED/BLUE first

        # ------------------- K-means iterations -------------------
        for iteration in range(MAX_ITERS):
            # Full assignments container
            full_assignments = [None] * len(points_coords)

            # Only show dotted line demo for first DEMO_ITERS iterations
            if iteration < DEMO_ITERS:
                # Choose 2 demo points: one from each cluster, rotating
                demo = []
                if alternate_order:
                    # Show B first, then A
                    if B_indices:
                        demo.append(B_indices[b_ptr])
                    if A_indices:
                        demo.append(A_indices[a_ptr])
                else:
                    # Show A first, then B
                    if A_indices:
                        demo.append(A_indices[a_ptr])
                    if B_indices:
                        demo.append(B_indices[b_ptr])

                # Advance pointers for next iteration (round-robin)
                if A_indices:
                    a_ptr = (a_ptr + 1) % len(A_indices)
                if B_indices:
                    b_ptr = (b_ptr + 1) % len(B_indices)
                alternate_order = not alternate_order

                # Demo: for each selected point, draw to red (pause), draw to blue (pause), remove, then color
                for idx in demo:
                    x, y = points_coords[idx]
                    p = axes.c2p(x, y)
                    r = red_centroid.get_center()
                    b = blue_centroid.get_center()

                    line_r = DashedLine(p, r, dash_length=DASH_LEN, color=PURE_RED,
                                        stroke_width=LINE_STROKE).set_z_index(Z_LINES)
                    self.play(ShowCreation(line_r), run_time=0.25)
                    self.wait(1.0)

                    line_b = DashedLine(p, b, dash_length=DASH_LEN, color=PURE_BLUE,
                                        stroke_width=LINE_STROKE).set_z_index(Z_LINES)
                    self.play(ShowCreation(line_b), run_time=0.25)
                    self.wait(1.0)

                    assign = "red" if dist(p, r) < dist(p, b) else "blue"
                    full_assignments[idx] = assign

                    self.play(Uncreate(line_r), Uncreate(line_b), run_time=0.25)
                    self.play(points[idx].animate.set_color(color_for(assign)), run_time=0.25)
                    self.wait(1.0)  # 1-second pause after each individual dot coloring

                # Assign remaining points (non-demo) and color them together
                for i in range(len(points_coords)):
                    if full_assignments[i] is not None:
                        continue
                    x, y = points_coords[i]
                    p = axes.c2p(x, y)
                    r = red_centroid.get_center()
                    b = blue_centroid.get_center()
                    full_assignments[i] = "red" if dist(p, r) < dist(p, b) else "blue"

                self.play(*[
                    points[i].animate.set_color(color_for(full_assignments[i]))
                    for i in range(len(points_coords)) if i not in demo
                ], run_time=0.9)
                self.wait(1.0)  # 1-second pause after batch coloring

            else:
                # After DEMO_ITERS: just assign colors without dotted line demonstrations
                for i in range(len(points_coords)):
                    x, y = points_coords[i]
                    p = axes.c2p(x, y)
                    r = red_centroid.get_center()
                    b = blue_centroid.get_center()
                    full_assignments[i] = "red" if dist(p, r) < dist(p, b) else "blue"

                # Color all points at once
                self.play(*[
                    points[i].animate.set_color(color_for(full_assignments[i]))
                    for i in range(len(points_coords))
                ], run_time=0.6)

            # Compute new centroids (scene coords) and check convergence
            red_pts = [axes.c2p(*points_coords[i]) for i, a in enumerate(full_assignments) if a == "red"]
            blue_pts = [axes.c2p(*points_coords[i]) for i, a in enumerate(full_assignments) if a == "blue"]

            if len(red_pts) > 0:
                red_mean = np.mean(np.vstack(red_pts), axis=0);
                red_mean[2] = 0
            else:
                red_mean = red_centroid.get_center()
            if len(blue_pts) > 0:
                blue_mean = np.mean(np.vstack(blue_pts), axis=0);
                blue_mean[2] = 0
            else:
                blue_mean = blue_centroid.get_center()

            r_old = red_centroid.get_center()
            b_old = blue_centroid.get_center()
            converged = (dist(r_old, red_mean) < TOL) and (dist(b_old, blue_mean) < TOL)

            if converged:
                break

            # Move centroids AND reset ALL dots to GREY for the next iteration
            self.play(
                red_centroid.animate.move_to(red_mean),
                red_X.animate.move_to(red_mean),
                blue_centroid.animate.move_to(blue_mean),
                blue_X.animate.move_to(blue_mean),
                *[pt.animate.set_color(GREY) for pt in points],
                run_time=1.1
            )
            self.wait(0.2)

        # ------------------- Final coloring -------------------
        final_assignments = []
        for i, (x, y) in enumerate(points_coords):
            p = axes.c2p(x, y)
            r = red_centroid.get_center()
            b = blue_centroid.get_center()
            final_assignments.append("red" if dist(p, r) < dist(p, b) else "blue")

        self.play(*[
            points[i].animate.set_color(color_for(final_assignments[i]))
            for i in range(len(points))
        ], run_time=0.8)

        # Fade out centroids; keep colored dots only
        self.play(
            FadeOut(red_centroid), FadeOut(red_X),
            FadeOut(blue_centroid), FadeOut(blue_X),
            run_time=0.7
        )
        self.wait(2)

        from scipy.spatial import ConvexHull

        # Separate points by final assignment
        red_cluster_points = []
        blue_cluster_points = []

        for i, assignment in enumerate(final_assignments):
            x, y = points_coords[i]
            screen_point = axes.c2p(x, y)
            if assignment == "red":
                red_cluster_points.append(screen_point[:2])  # Only x,y coordinates
            else:
                blue_cluster_points.append(screen_point[:2])

        # Create convex hull shapes
        shapes_to_add = []

        if len(red_cluster_points) >= 3:  # Need at least 3 points for ConvexHull
            red_points_array = np.array(red_cluster_points)
            red_hull = ConvexHull(red_points_array)
            red_vertices = red_points_array[red_hull.vertices]
            # Convert back to 3D points for Manim
            red_vertices_3d = [np.array([x, y, 0]) for x, y in red_vertices]
            red_shape = Polygon(*red_vertices_3d,
                                fill_color=RED, fill_opacity=0.2,
                                stroke_width=0).set_z_index(Z_DOTS - 1).scale(1.)  # Behind dots
            shapes_to_add.append(red_shape)

        if len(blue_cluster_points) >= 3:  # Need at least 3 points for ConvexHull
            blue_points_array = np.array(blue_cluster_points)
            blue_hull = ConvexHull(blue_points_array)
            blue_vertices = blue_points_array[blue_hull.vertices]
            # Convert back to 3D points for Manim
            blue_vertices_3d = [np.array([x, y, 0]) for x, y in blue_vertices]
            blue_shape = Polygon(*blue_vertices_3d,
                                 fill_color=BLUE, fill_opacity=0.2,
                                 stroke_width=0).set_z_index(Z_DOTS - 1).scale(1.3)  # Behind dots
            shapes_to_add.append(blue_shape)

        # Animate the shapes appearing
        if shapes_to_add:
            self.play(*[FadeIn(shape) for shape in shapes_to_add], run_time=1.0)

        self.wait(2)


class kmeansMath(Scene):

    def construct(self):
        datapoint = Tex(r"X = \{x^{(1)}, x^{(2)}, x^{(3)}, \dots, x^{(n)}\}, \; \ x^{(i)} \in \mathbb{R}^d").to_edge(
            UP).shift(DOWN * 0.66).set_color_by_gradient(RED, ORANGE, BLUE)

        self.play(ShowCreation(datapoint))
        self.wait(2)

        centroids = Tex(r"\mu_1, \ \mu_2, \ \mu_3, \ \dots, \ \mu_K").next_to(datapoint, DOWN).shift(DOWN * 0.5).scale(
            1.2).set_color_by_gradient(RED, ORANGE, BLUE)
        self.play(ShowCreation(centroids))
        self.wait(2)

        self.play(Transform(centroids, Tex(r"\mu_1^{(0)}, \ \mu_2^{(0)}, \ \mu_3^{(0)}, \ \dots, \ \mu_K^{(0)}").scale(
            1.02).move_to(centroids).set_color_by_gradient(RED, ORANGE, BLUE)))
        self.wait(2)

        assignment = Tex(
            r"c_i = \arg\min_{k \in \{1,\dots,K\}} \left\lVert x^{(i)} - \mu_k^{(0)} \right\rVert^2").next_to(centroids,
                                                                                                              DOWN).shift(
            DOWN * 0.5).scale(1.2).set_color_by_gradient(RED, ORANGE, BLUE)

        self.play(ShowCreation(assignment))
        self.wait(2)

        brace = Brace(assignment[-16:], DOWN).shift(DOWN * 0.15).set_color(YELLOW)
        self.play(ShowCreation(brace))
        self.wait(2)

        self.play(Transform(brace, Brace(assignment[-30:-17], DOWN).shift(DOWN * 0.15).set_color(YELLOW)))
        self.wait(2)

        self.play(Transform(brace, SurroundingRectangle(assignment[:2]).scale(1.12)))
        self.wait(2)

        self.play(FadeOut(brace))

        update = Tex(r"\mu_k^{(t+1)} = \frac{1}{|C_k|} \sum_{i \in C_k} x^{(i)}").next_to(assignment, DOWN).shift(
            DOWN * 0.76).scale(1.2).set_color_by_gradient(RED, ORANGE, BLUE)

        self.play(ShowCreation(update))
        self.play(self.camera.frame.animate.shift(DOWN * 2.2).scale(0.8),
                  FadeOut(Group(datapoint, centroids, assignment)))

        rect = SurroundingRectangle(update[:7], color=YELLOW).scale(1.12)
        self.play(ShowCreation(rect))
        self.wait(2)

        self.play(Transform(rect, SurroundingRectangle(update[8:], color=YELLOW).scale(1.02)))
        self.wait(2)

        a = Tex(r"d(x, \mu) = \sqrt{(x_1 - \mu_1)^2 + (x_2 - \mu_2)^2}").scale(1.12).shift(
            RIGHT * 18 + DOWN * 2.22).set_color_by_gradient(RED, ORANGE, BLUE)

        self.play(self.camera.frame.animate.shift(RIGHT * 18), ShowCreation(a))
        self.wait(2)

        self.play(Transform(a, Tex(r"d(x, \mu) = (x_1 - \mu_1)^2 + (x_2 - \mu_2)^2").scale(1.22).move_to(
            a).set_color_by_gradient(RED, ORANGE, BLUE)))

        self.wait(2)

        self.play(a.animate.shift(UP * 0.94))

        b = Tex(
            r"\arg\min_k \lVert x^{(i)} - \mu_k \rVert \ = \ \arg\min_k \lVert x^{(i)} - \mu_k \rVert^2").set_color_by_gradient(
            RED, ORANGE, BLUE)

        b.next_to(a, DOWN).shift(DOWN * 1.3)

        self.play(ShowCreation(b))
        self.wait(2)

        rect = SurroundingRectangle(b[:16], color=YELLOW).scale(1.02)
        rect1 = SurroundingRectangle(b[17:], color=YELLOW).scale(1.02)

        self.play(ShowCreation(rect))
        self.play(ShowCreation(rect1))
        self.wait(2)

        self.play(FadeOut(Group(rect, rect1)), b.animate.shift(UP * 0.52), self.camera.frame.animate.shift(DOWN * 0.7))
        self.wait(2)

        c = Tex(r"if \ a < b \Rightarrow \sqrt{a} < \sqrt{b}").scale(1.2).set_color_by_gradient(RED, ORANGE, BLUE)
        c.next_to(b, DOWN).shift(DOWN * 0.4)

        self.play(ShowCreation(c))
        self.wait(2)

        self.embed()


class ChoosingK(Scene):
    def construct(self):
        # Create axes
        axes = SimpleAxes(
            x_max=10, y_max=7,
            x_length=8.8, y_length=5.2,
            origin=ORIGIN + 2.5 * DOWN + 4.4 * LEFT,
            stroke_width=2, tip=True
        )

        x_label = Tex("x_{1}")
        y_label = Tex("x_{2}")
        x_label.next_to(axes.x_axis.get_end(), DOWN, buff=0.28)
        y_label.next_to(axes.y_axis.get_end(), LEFT, buff=0.28)

        axes_group = VGroup(axes, x_label, y_label).scale(1.17)
        self.play(ShowCreation(axes_group))

        # Create dispersed dataset shifted UP slightly from previous version
        base_dataset = [
            # Left region - shifted up a bit
            (0.8, 2.9), (1.9, 2.5), (0.5, 3.5), (1.4, 3.1), (1.8, 2.2),
            (0.6, 3.8), (2.3, 2.7), (1.2, 3.4), (0.8, 2.6), (2.1, 3.2),
            (1.1, 4.2), (1.7, 4.5), (0.9, 3.9), (2.0, 4.1), (0.4, 4.4),

            # Center-top region - shifted up
            (3.8, 4.9), (4.9, 4.5), (3.5, 4.2), (5.1, 4.8), (3.6, 5.4),
            (4.2, 4.3), (4.0, 5.1), (4.8, 4.7), (4.5, 4.0), (3.7, 4.7),

            # Center-bottom region - shifted up
            (4.6, 2.5), (3.9, 2.1), (4.8, 2.9), (3.7, 2.4), (5.0, 2.2),
            (4.1, 3.1), (4.9, 1.8), (3.8, 2.7), (4.3, 2.3), (4.2, 3.0),

            # Right region - shifted up
            (7.2, 1.9), (8.6, 2.5), (7.0, 1.6), (8.2, 2.2), (8.1, 2.8),
            (8.7, 1.7), (7.4, 2.4), (8.4, 2.0), (7.8, 2.6), (8.5, 1.8),
            (6.9, 3.9), (7.9, 4.6), (7.1, 3.5), (8.3, 4.3), (7.5, 3.7),
            (8.4, 4.5), (6.8, 4.2), (8.0, 3.4), (8.2, 4.1), (7.2, 3.6),
        ]

        # Create dots
        dots = VGroup(*[
            Dot(point=axes.c2p(x, y), radius=0.12, color=GREY).scale(0.7)
            for x, y in base_dataset
        ])

        self.play(FadeIn(dots, lag_ratio=0.05), run_time=2)
        self.wait(1)

        # Colors for different K values
        colors_k2 = [RED, BLUE]
        colors_k3 = [RED, GREEN, BLUE]
        colors_k4 = [RED, GREEN, BLUE, YELLOW]
        colors_k5 = [RED, GREEN, BLUE, YELLOW, PURPLE]


        # Function to create convex hull shapes - NO STROKE, ONLY FILL
        def create_cluster_shapes(points_coords, labels, colors, axes):
            shapes = []
            unique_labels = list(set(labels))

            for i, label in enumerate(unique_labels):
                cluster_points = []
                for j, point_label in enumerate(labels):
                    if point_label == label:
                        x, y = points_coords[j]
                        screen_point = axes.c2p(x, y)
                        cluster_points.append(screen_point[:2])

                if len(cluster_points) >= 3:
                    points_array = np.array(cluster_points)
                    try:
                        hull = ConvexHull(points_array)
                        vertices = points_array[hull.vertices]
                        vertices_3d = [np.array([x, y, 0]) for x, y in vertices]

                        # NO STROKE - only fill, scaled by 1.30
                        shape = Polygon(*vertices_3d,
                                        fill_color=colors[i], fill_opacity=0.25,
                                        stroke_width=0).scale(1.30).set_z_index(-1)
                        shapes.append(shape)
                    except:
                        continue

            return shapes


        k_values = [2, 3, 4, 5]
        all_colors = [colors_k2, colors_k3, colors_k4, colors_k5]

        # K label - larger scale and positioned DOWN*0.5 + LEFT*0.5
        k_label = None
        final_shapes = None  # Keep track of final shapes for question marks

        for idx, k in enumerate(k_values):
            # Create K label with larger scale and shifted DOWN*0.5 + LEFT*0.5
            new_k_label = Text(f"K = {k}", font_size=48, weight=BOLD, color=WHITE)
            new_k_label.scale(1.6)  # Larger scale
            new_k_label.to_edge(UP + RIGHT, buff=0.8)
            new_k_label.shift(DOWN * 0.5 + LEFT * 0.5)  # Shift as requested

            if idx == 0:
                k_label = new_k_label
                self.play(Write(k_label))
            else:
                # Transform the existing label to new K value
                self.play(Transform(k_label, new_k_label))

            # Use scikit-learn KMeans to cluster the data
            data_array = np.array(base_dataset)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data_array)

            # Color the dots based on clustering
            color_animations = []
            for i, label in enumerate(labels):
                color = all_colors[idx][label]
                color_animations.append(dots[i].animate.set_color(color))

            self.play(*color_animations, run_time=1.5)
            self.wait(0.5)

            # Create and show cluster shapes
            shapes = create_cluster_shapes(base_dataset, labels, all_colors[idx], axes)
            if shapes:
                self.play(*[FadeIn(shape) for shape in shapes], run_time=1.0)
                self.wait(2)

                # Store final shapes for question marks
                if idx == len(k_values) - 1:
                    final_shapes = shapes
                    final_labels = labels
                    final_colors = all_colors[idx]

                # Remove shapes before next iteration (except for last one)
                if idx < len(k_values) - 1:
                    self.play(*[FadeOut(shape) for shape in shapes], run_time=0.5)
                    # Reset dots to grey
                    self.play(*[dot.animate.set_color(GREY) for dot in dots], run_time=0.5)

        self.wait(2)


class ElbowMethod(Scene):
    def construct(self):
        # REDUCED RANGES to bring graph closer to axes

        self.camera.frame.scale(1.2).shift(UP * 0.6)

        axes = SimpleAxes(
            x_max=9.5, y_max=12,  # Reduced from x_max=10, y_max=16
            x_length=9, y_length=6,
            origin=ORIGIN + 2.5 * DOWN + 4.4 * LEFT,
            stroke_width=3, tip=True,
            include_ticks=True,
            tick_size=0.12
        ).set_color(WHITE)

        # Labels
        x_label = Tex("K", font_size=48).next_to(axes.x_axis.get_end(), DOWN, buff=0.3)
        y_label = Tex("J", font_size=48).next_to(axes.y_axis.get_end(), LEFT, buff=0.3)

        axes_group = VGroup(axes, x_label, y_label).scale(1.17)
        self.play(ShowCreation(axes_group), run_time=2)
        self.wait(1)

        # ADJUSTED elbow method data - scaled to fit better in the new range
        k_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        j_values = np.array([11.5, 8.2, 5.8, 4.5, 3.8, 3.4, 3.1, 2.95, 2.85])

        # Create points using your axes.c2p() method - COLORED YELLOW
        points = VGroup(*[
            Dot(axes.c2p(k, j), radius=0.12, color=YELLOW)
            for k, j in zip(k_values, j_values)
        ])

        # Set all dots to yellow color
        for dot in points:
            dot.set_color(YELLOW).shift(DOWN)

        # Create curve by connecting points
        curve_points = []
        for i in range(len(k_values)):
            curve_points.append(axes.c2p(k_values[i], j_values[i]))

        curve = VMobject()
        curve.set_points_smoothly(curve_points)
        curve.set_stroke(BLUE, width=4).shift(DOWN)

        # Animate the curve and points
        self.play(ShowCreation(curve), run_time=2.5)
        self.play(FadeIn(points, lag_ratio=0.15), run_time=2)
        self.wait(1)

        # Highlight the elbow point (K=3)
        elbow_k = 3
        elbow_j = j_values[elbow_k - 1]  # K=3 is at index 2
        elbow_point = axes.c2p(elbow_k, elbow_j)

        # Create highlighted elbow point
        elbow_dot = Dot(elbow_point, radius=0.18, color=RED, stroke_width=3).shift(DOWN).set_color(RED)
        elbow_circle = Circle(radius=0.35, color=RED, stroke_width=3).move_to(elbow_dot)

        # Arrow pointing to elbow point
        arrow_start = elbow_point + UP * 2.5 + RIGHT * 1.5
        arrow = Arrow(
            start=arrow_start,
            end=elbow_point + UP * 0.3 + RIGHT * 0.2,
            buff=0,
            color=RED,
            stroke_width=4,
        ).set_color(RED).shift(RIGHT * 0.1 + DOWN * 0.88)

        # "Elbow Point" text
        elbow_text = Text("Elbow Point", font_size=56, weight=BOLD, color=RED)
        elbow_text.next_to(arrow_start, UP, buff=0.14).shift(DOWN * 0.5)

        # Animate elbow highlighting
        self.play(
            FadeIn(elbow_dot),
            ShowCreation(elbow_circle),
            run_time=1.5
        )
        self.wait(0.5)

        self.play(
            GrowArrow(arrow),
            Write(elbow_text),
            run_time=1.5
        )

        # Add K=3 label
        k3_label = Text("K = 3", font_size=62, color=RED, weight=BOLD)
        k3_label.next_to(elbow_point, DOWN + RIGHT, buff=1.2).shift(UP * 1.2)

        self.play(Write(k3_label), run_time=1)
        self.wait(3)


class CitiesExample(Scene):
    def construct(self):
        self.camera.frame.scale(1.05)
        # Create axes
        axes = SimpleAxes(
            x_max=5000, y_max=80,
            x_length=8.8, y_length=5.2,
            origin=ORIGIN + 2.5 * DOWN + 4.4 * LEFT,
            stroke_width=2, tip=True
        )

        x_label = Tex("Population")
        y_label = Tex("GDP")
        x_label.next_to(axes.x_axis.get_end(), DOWN, buff=0.28)
        y_label.next_to(axes.y_axis.get_end(), LEFT, buff=0.28)

        a = VGroup(axes, x_label, y_label).scale(1.17)
        self.play(ShowCreation(a))

        # Create cleaner scattered data with clear patterns (shifted DOWN)
        np.random.seed(42)
        cities_data = []

        # Shift all GDP values down by 12 to move datapoints lower
        shift_down = 12

        # Generate cleaner data clusters with less noise
        # Cluster 1: Rural (low population, low GDP) - shifted down
        for i in range(8):
            x = np.random.uniform(100, 600)
            y = np.random.uniform(22, 35) - shift_down
            cities_data.append((x, y))

        # Cluster 2: Small (low-medium population, medium GDP) - shifted down
        for i in range(10):
            x = np.random.uniform(600, 1800)
            y = np.random.uniform(35, 50) - shift_down
            cities_data.append((x, y))

        # Cluster 3: Medium (medium population, higher GDP) - shifted down
        for i in range(12):
            x = np.random.uniform(1800, 3200)
            y = np.random.uniform(50, 65) - shift_down
            cities_data.append((x, y))

        # Cluster 4: Large (high population, high GDP) - shifted down
        for i in range(10):
            x = np.random.uniform(3200, 4200)
            y = np.random.uniform(62, 75) - shift_down
            cities_data.append((x, y))

        # Cluster 5: Mega (very high population, very high GDP) - shifted down
        for i in range(8):
            x = np.random.uniform(4200, 4800)
            y = np.random.uniform(70, 78) - shift_down
            cities_data.append((x, y))

        # Create dots
        dots = VGroup(*[
            Dot(point=axes.c2p(x, y), radius=0.12, color=GREY).scale(0.7)
            for x, y in cities_data
        ])

        self.play(FadeIn(dots, lag_ratio=0.05), run_time=2)
        self.wait(1)

        # Show K=3 text first - SCALED BY 2
        k3_label = Text("K = 3", font_size=48, weight=BOLD).to_edge(UP).shift(DOWN * 0.5).scale(2)
        self.play(Write(k3_label))
        self.wait(1)

        # K=3 clustering
        data_array = np.array(cities_data)
        kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_3 = kmeans_3.fit_predict(data_array)

        k3_colors = [BLUE, GREEN, RED]
        k3_names = ["SMALL", "MEDIUM", "LARGE"]

        # Color dots for K=3
        color_animations_3 = []
        for i, label in enumerate(labels_3):
            color_animations_3.append(dots[i].animate.set_color(k3_colors[label]))

        self.play(*color_animations_3, run_time=1.5)
        self.wait(1)

        # Create convex hull shapes for K=3
        shapes_3 = []

        for cluster_id in range(3):
            cluster_points = []
            for i, label in enumerate(labels_3):
                if label == cluster_id:
                    x, y = cities_data[i]
                    screen_point = axes.c2p(x, y)
                    cluster_points.append(screen_point[:2])

            if len(cluster_points) >= 3:
                points_array = np.array(cluster_points)
                hull = ConvexHull(points_array)
                vertices = points_array[hull.vertices]
                vertices_3d = [np.array([x, y, 0]) for x, y in vertices]

                # Convex hull with only fill, no stroke, scaled by 1.3
                shape = Polygon(*vertices_3d,
                                fill_color=k3_colors[cluster_id], fill_opacity=0.3,
                                stroke_width=0).scale(1.3).set_z_index(-1)
                shapes_3.append(shape)

        # Show convex hulls
        self.play(*[FadeIn(shape) for shape in shapes_3], run_time=1.0)

        # Add cluster labels ABOVE (next_to UP) their respective convex hulls
        cluster_labels_3 = []
        for i, (color, name, shape) in enumerate(zip(k3_colors, k3_names, shapes_3)):
            label = Text(name, color=color, font_size=24, weight=BOLD).scale(2).set_color(color)
            label.next_to(shape, UP, buff=0.2)
            cluster_labels_3.append(label)

        self.play(*[Write(label) for label in cluster_labels_3], FadeOut(k3_label), )
        self.wait(2)

        # Transition to K=5
        self.play(

            *[FadeOut(label) for label in cluster_labels_3],
            *[FadeOut(shape) for shape in shapes_3]
        )

        # Show K=5 text first - SCALED BY 2
        k5_label = Text("K = 5", font_size=48, weight=BOLD).to_edge(UP).shift(DOWN * 0.5).scale(2)
        self.play(Write(k5_label))
        self.wait(1)

        # K=5 clustering
        kmeans_5 = KMeans(n_clusters=5, random_state=42, n_init=10)
        labels_5 = kmeans_5.fit_predict(data_array)

        k5_colors = [PURPLE, BLUE, GREEN, RED, ORANGE]
        k5_names = ["MEGA", "SMALL", "MEDIUM", "RURAL", "LARGE"]

        # Color dots for K=5
        color_animations_5 = []
        for i, label in enumerate(labels_5):
            color_animations_5.append(dots[i].animate.set_color(k5_colors[label]))

        self.play(*color_animations_5, run_time=1.5)
        self.wait(1)

        # Create convex hull shapes for K=5
        shapes_5 = []

        for cluster_id in range(5):
            cluster_points = []
            for i, label in enumerate(labels_5):
                if label == cluster_id:
                    x, y = cities_data[i]
                    screen_point = axes.c2p(x, y)
                    cluster_points.append(screen_point[:2])

            if len(cluster_points) >= 3:
                points_array = np.array(cluster_points)
                hull = ConvexHull(points_array)
                vertices = points_array[hull.vertices]
                vertices_3d = [np.array([x, y, 0]) for x, y in vertices]

                # Convex hull with only fill, no stroke, scaled by 1.3
                shape = Polygon(*vertices_3d,
                                fill_color=k5_colors[cluster_id], fill_opacity=0.3,
                                stroke_width=0).scale(1.3).set_z_index(-1)
                shapes_5.append(shape)

        # Show convex hulls
        self.play(*[FadeIn(shape) for shape in shapes_5], run_time=1.0)

        # Add cluster labels ABOVE (next_to UP) their respective convex hulls
        cluster_labels_5 = []
        for i, (color, name, shape) in enumerate(zip(k5_colors, k5_names, shapes_5)):
            label = Text(name, color=color, font_size=20, weight=BOLD).scale(2).set_color(color)
            label.next_to(shape, UP, buff=0.34)
            cluster_labels_5.append(label)

        self.play(*[Write(label) for label in cluster_labels_5], FadeOut(k5_label))
        self.wait(3)


class CitiesKMeansWithInitialization(Scene):
    def construct(self):
        self.camera.frame.scale(1.08)

        # Tunables
        DOT_RADIUS = 0.12
        CENTROID_RADIUS = 0.25
        CENTROID_STROKE = 4
        MAX_ITERS = 8
        TOL = 1e-6

        # Colors
        DOT_RED = RED
        DOT_GREEN = GREEN
        DOT_YELLOW = YELLOW
        PURE_RED = "#FF0000"
        PURE_GREEN = "#00FF00"
        PURE_YELLOW = "#FFFF00"

        # Z-index layering
        Z_DOTS = 0
        Z_CENTROIDS = 3
        Z_CENTROID_LABELS = 4


        # Helper functions
        def color_for(assign):
            if assign == "red":
                return DOT_RED
            elif assign == "green":
                return DOT_GREEN
            else:
                return DOT_YELLOW


        def dist(p, q):
            return np.linalg.norm(p - q)


        # Create axes
        axes = SimpleAxes(
            x_max=5000, y_max=80,
            x_length=8.8, y_length=5.2,
            origin=ORIGIN + 2.5 * DOWN + 4.4 * LEFT,
            stroke_width=2, tip=True
        )

        x_label = Tex("Population")
        y_label = Tex("GDP")
        x_label.next_to(axes.x_axis.get_end(), DOWN, buff=0.28)
        y_label.next_to(axes.y_axis.get_end(), LEFT, buff=0.28)

        a = VGroup(axes, x_label, y_label).scale(1.17)
        self.play(ShowCreation(a))

        # Create cities data (shifted down)
        np.random.seed(42)
        cities_data = []
        shift_down = 12

        # Generate data clusters
        for i in range(8):
            x = np.random.uniform(100, 600)
            y = np.random.uniform(22, 35) - shift_down
            cities_data.append((x, y))

        for i in range(10):
            x = np.random.uniform(600, 1800)
            y = np.random.uniform(35, 50) - shift_down
            cities_data.append((x, y))

        for i in range(12):
            x = np.random.uniform(1800, 3200)
            y = np.random.uniform(50, 65) - shift_down
            cities_data.append((x, y))

        for i in range(10):
            x = np.random.uniform(3200, 4200)
            y = np.random.uniform(62, 75) - shift_down
            cities_data.append((x, y))

        for i in range(8):
            x = np.random.uniform(4200, 4800)
            y = np.random.uniform(70, 78) - shift_down
            cities_data.append((x, y))

        # Create dots
        dots = VGroup(*[
            Dot(point=axes.c2p(x, y), radius=DOT_RADIUS, color=GREY).scale(0.7).set_z_index(Z_DOTS)
            for x, y in cities_data
        ])

        self.play(FadeIn(dots, lag_ratio=0.02, run_time=1.5))
        self.wait(1)

        # INITIALIZATION: Select 3 points for moderate iterations
        selected_indices = [5, 25, 40]  # From different clusters for good spread

        # Create smaller circles around selected dots (scaled by 0.5)
        init_circles = []
        circle_colors = [PURE_RED, PURE_GREEN, PURE_YELLOW]

        for i, (dot_idx, color) in enumerate(zip(selected_indices, circle_colors)):
            circle = Circle(radius=0.4, stroke_color=color, stroke_width=6, fill_opacity=0).scale(0.5)
            circle.move_to(dots[dot_idx].get_center())  # Move to exact dot position
            init_circles.append(circle)

        # Show circles around selected dots
        self.play(*[ShowCreation(circle) for circle in init_circles], run_time=1.5)
        self.wait(2)

        # Create centroids at EXACT same positions as the selected dots (inside circles)
        red_centroid = Circle(
            radius=CENTROID_RADIUS,
            stroke_color=PURE_RED, fill_color=PURE_RED,
            stroke_width=CENTROID_STROKE, fill_opacity=1.0
        ).move_to(dots[selected_indices[0]].get_center()).set_z_index(Z_CENTROIDS)  # EXACT position

        green_centroid = Circle(
            radius=CENTROID_RADIUS,
            stroke_color=PURE_GREEN, fill_color=PURE_GREEN,
            stroke_width=CENTROID_STROKE, fill_opacity=1.0
        ).move_to(dots[selected_indices[1]].get_center()).set_z_index(Z_CENTROIDS)  # EXACT position

        yellow_centroid = Circle(
            radius=CENTROID_RADIUS,
            stroke_color=PURE_YELLOW, fill_color=PURE_YELLOW,
            stroke_width=CENTROID_STROKE, fill_opacity=1.0
        ).move_to(dots[selected_indices[2]].get_center()).set_z_index(Z_CENTROIDS)  # EXACT position

        # Create larger X marks inside centroids
        red_X = Text("×", weight=BOLD, font_size=42).move_to(red_centroid.get_center()).set_color(WHITE).set_z_index(
            Z_CENTROID_LABELS)
        green_X = Text("×", weight=BOLD, font_size=42).move_to(green_centroid.get_center()).set_color(
            WHITE).set_z_index(Z_CENTROID_LABELS)
        yellow_X = Text("×", weight=BOLD, font_size=42).move_to(yellow_centroid.get_center()).set_color(
            BLACK).set_z_index(Z_CENTROID_LABELS)  # BLACK text for yellow

        red_centroid.add(red_X)
        green_centroid.add(green_X)
        yellow_centroid.add(yellow_X)

        red_centroid.move_to(init_circles[0])
        green_centroid.move_to(init_circles[1])
        yellow_centroid.move_to(init_circles[2])

        centroids = [red_centroid, green_centroid, yellow_centroid]
        centroid_Xs = [red_X, green_X, yellow_X]
        centroid_colors = [DOT_RED, DOT_GREEN, DOT_YELLOW]

        # Fade in centroids (INSIDE the circles), fade out circles
        self.play(
            FadeIn(Group(red_centroid, green_centroid, yellow_centroid)),
            *[FadeOut(circle) for circle in init_circles],
            run_time=2
        )
        self.wait(1)

        # K-means iterations (NO DOTTED LINES)
        for iteration in range(MAX_ITERS):
            # Assign all points to nearest centroid (no line demonstrations)
            assignments = []
            for i, (x, y) in enumerate(cities_data):
                p = axes.c2p(x, y)
                distances = [dist(p, c.get_center()) for c in centroids]
                closest_idx = distances.index(min(distances))
                assignments.append(["red", "green", "yellow"][closest_idx])

            # Color all dots based on assignments
            color_animations = []
            for i, assignment in enumerate(assignments):
                color_animations.append(dots[i].animate.set_color(color_for(assignment)))

            self.play(*color_animations, run_time=0.77)
            self.wait(0.333)

            # Compute new centroids
            cluster_points = {"red": [], "green": [], "yellow": []}
            for i, assignment in enumerate(assignments):
                x, y = cities_data[i]
                cluster_points[assignment].append(axes.c2p(x, y))

            new_positions = []
            for cluster_name in ["red", "green", "yellow"]:
                if len(cluster_points[cluster_name]) > 0:
                    mean_pos = np.mean(np.vstack(cluster_points[cluster_name]), axis=0)
                    mean_pos[2] = 0  # Zero z-coordinate
                    new_positions.append(mean_pos)
                else:
                    # Keep current position if no points assigned
                    current_idx = ["red", "green", "yellow"].index(cluster_name)
                    new_positions.append(centroids[current_idx].get_center())

            # Check convergence
            old_positions = [c.get_center() for c in centroids]
            converged = all(dist(old_pos, new_pos) < TOL
                            for old_pos, new_pos in zip(old_positions, new_positions))

            if converged:
                break

            # Move centroids and reset dots to grey for next iteration
            move_animations = []
            for i, new_pos in enumerate(new_positions):
                move_animations.append(centroids[i].animate.move_to(new_pos))
                move_animations.append(centroid_Xs[i].animate.move_to(new_pos))

            reset_animations = [dot.animate.set_color(GREY) for dot in dots]

            self.play(*move_animations, *reset_animations, run_time=1.2)
            self.wait(0.33)

        # Final assignment and coloring
        final_assignments = []
        for i, (x, y) in enumerate(cities_data):
            p = axes.c2p(x, y)
            distances = [dist(p, c.get_center()) for c in centroids]
            closest_idx = distances.index(min(distances))
            final_assignments.append(["red", "green", "yellow"][closest_idx])

        final_colors = []
        for i, assignment in enumerate(final_assignments):
            final_colors.append(dots[i].animate.set_color(color_for(assignment)))

        self.play(*final_colors, run_time=1.0)

        # Remove centroids
        self.play(FadeOut(Group(red_centroid, green_centroid, yellow_centroid)), run_time=0.8)
        self.wait(1)

        # ADD CONVEX HULL FILLS AT THE END
        shapes = []

        for cluster_id, cluster_color in enumerate([DOT_RED, DOT_GREEN, DOT_YELLOW]):
            cluster_points = []
            cluster_name = ["red", "green", "yellow"][cluster_id]

            for i, assignment in enumerate(final_assignments):
                if assignment == cluster_name:
                    x, y = cities_data[i]
                    screen_point = axes.c2p(x, y)
                    cluster_points.append(screen_point[:2])

            if len(cluster_points) >= 3:
                points_array = np.array(cluster_points)
                hull = ConvexHull(points_array)
                vertices = points_array[hull.vertices]
                vertices_3d = [np.array([x, y, 0]) for x, y in vertices]

                # Convex hull with only fill, no stroke, scaled by 1.3
                shape = Polygon(*vertices_3d,
                                fill_color=cluster_color, fill_opacity=0.3,
                                stroke_width=0).scale(1.3).set_z_index(-1)
                shapes.append(shape)

        # Show convex hulls
        if shapes:
            self.play(*[FadeIn(shape) for shape in shapes], run_time=1.5)

        self.wait(3)


class GridKMeansComparison(Scene):

    def construct(self):

        self.camera.frame.shift(UP * 0.96 + +RIGHT * 2)

        # Tunables
        DOT_RADIUS = 0.08
        CENTROID_RADIUS = 0.15
        CENTROID_STROKE = 3
        MAX_ITERS = 10
        TOL = 1e-6

        # Colors
        DOT_RED = RED
        DOT_GREEN = GREEN
        DOT_YELLOW = YELLOW
        PURE_RED = "#FF0000"
        PURE_GREEN = "#00FF00"
        PURE_YELLOW = "#FFFF00"

        # Grid positions
        positions = [
            LEFT * 3.0 + UP * 2,  # Top-left
            RIGHT * 3.0 + UP * 2,  # Top-right
            LEFT * 3.0 + DOWN * 2,  # Bottom-left
            RIGHT * 3.0 + DOWN * 2  # Bottom-right
        ]


        # Helper functions
        def color_for(assign):
            if assign == "red":
                return DOT_RED
            elif assign == "green":
                return DOT_GREEN
            else:
                return DOT_YELLOW


        def dist(p, q):
            return np.linalg.norm(p - q)


        def calculate_loss(assignments, centroids, data, axes):
            """Calculate Within-Cluster Sum of Squares (WCSS)"""
            total_loss = 0
            for i, (x, y) in enumerate(data):
                point_pos = axes.c2p(x, y)
                cluster_idx = ["red", "green", "yellow"].index(assignments[i])
                centroid_pos = centroids[cluster_idx].get_center()
                total_loss += dist(point_pos, centroid_pos) ** 2
            return total_loss


        # Create NEW "S-curve snake" dataset - produces VERY different losses
        np.random.seed(2024)
        cities_data = []

        # Create an S-shaped curve that can be clustered in dramatically different ways
        # Bottom-left dense cluster
        for i in range(18):
            x = np.random.uniform(800, 1200)
            y = np.random.uniform(20, 30)
            x += np.random.normal(0, 60)
            y += np.random.normal(0, 3)
            cities_data.append((max(600, min(3500, x)), max(15, min(65, y))))

        # S-curve: flowing from bottom-left, up to middle-top, then down to bottom-right
        # Middle transition region (creates major ambiguity)
        for i in range(22):
            t = i / 21.0  # 0 to 1
            x = 1200 + t * 800  # 1200 to 2000
            y = 30 + np.sin(t * np.pi * 2) * 20  # Oscillating height
            x += np.random.normal(0, 80)
            y += np.random.normal(0, 4)
            cities_data.append((max(600, min(3500, x)), max(15, min(65, y))))

        # Top dense cluster
        for i in range(16):
            x = np.random.uniform(2000, 2400)
            y = np.random.uniform(50, 60)
            x += np.random.normal(0, 70)
            y += np.random.normal(0, 3)
            cities_data.append((max(600, min(3500, x)), max(15, min(65, y))))

        # Final curve down to bottom-right
        for i in range(20):
            t = i / 19.0  # 0 to 1
            x = 2400 + t * 800  # 2400 to 3200
            y = 55 - t * 35  # 55 to 20
            x += np.random.normal(0, 90)
            y += np.random.normal(0, 4)
            cities_data.append((max(600, min(3500, x)), max(15, min(65, y))))

        # Bottom-right dense cluster
        for i in range(14):
            x = np.random.uniform(3000, 3400)
            y = np.random.uniform(18, 28)
            x += np.random.normal(0, 60)
            y += np.random.normal(0, 3)
            cities_data.append((max(600, min(3500, x)), max(15, min(65, y))))

        # STRATEGIC initializations designed to produce VERY different final losses
        initialization_scenarios = [
            [2, 20, 45],  # Natural end points + middle - should produce LOW loss
            [8, 12, 16],  # All clustered at start - should produce HIGH loss
            [55, 60, 65],  # All clustered at end - should produce HIGH loss
            [25, 35, 50]  # Mixed strategic - should produce MEDIUM loss
        ]

        # Create all 4 grid elements
        all_axes = []
        all_dots = []
        all_centroids = []
        all_centroid_Xs = []

        for grid_idx, (pos, selected_indices) in enumerate(zip(positions, initialization_scenarios)):
            # Create axes for this grid position
            axes = SimpleAxes(
                x_max=4000, y_max=70,
                x_length=4.0, y_length=2.5,
                origin=pos,
                stroke_width=2, tip=True
            )
            all_axes.append(axes)

            # Create dots for this grid
            dots = VGroup(*[
                Dot(point=axes.c2p(x, y), radius=DOT_RADIUS, color=GREY).scale(0.6)
                for x, y in cities_data
            ])
            all_dots.append(dots)

            # Create centroids at selected positions
            red_centroid = Circle(
                radius=CENTROID_RADIUS,
                stroke_color=PURE_RED, fill_color=PURE_RED,
                stroke_width=CENTROID_STROKE, fill_opacity=1.0
            ).move_to(dots[selected_indices[0]].get_center())

            green_centroid = Circle(
                radius=CENTROID_RADIUS,
                stroke_color=PURE_GREEN, fill_color=PURE_GREEN,
                stroke_width=CENTROID_STROKE, fill_opacity=1.0
            ).move_to(dots[selected_indices[1]].get_center())

            yellow_centroid = Circle(
                radius=CENTROID_RADIUS,
                stroke_color=PURE_YELLOW, fill_color=PURE_YELLOW,
                stroke_width=CENTROID_STROKE, fill_opacity=1.0
            ).move_to(dots[selected_indices[2]].get_center())

            # Create X marks
            red_X = Text("×", weight=BOLD, font_size=32).move_to(red_centroid.get_center()).set_color(WHITE)
            green_X = Text("×", weight=BOLD, font_size=32).move_to(green_centroid.get_center()).set_color(WHITE)
            yellow_X = Text("×", weight=BOLD, font_size=32).move_to(yellow_centroid.get_center()).set_color(BLACK)

            red_centroid.add(red_X)
            green_centroid.add(green_X)
            yellow_centroid.add(yellow_X)

            centroids = [red_centroid, green_centroid, yellow_centroid]
            centroid_Xs = [red_X, green_X, yellow_X]

            all_centroids.append(centroids)
            all_centroid_Xs.append(centroid_Xs)

        # Show all axes and dots simultaneously
        self.play(*[ShowCreation(axes) for axes in all_axes])
        self.play(*[FadeIn(dots, lag_ratio=0.01) for dots in all_dots], run_time=2)

        # Show all initial centroids
        self.play(*[FadeIn(Group(*centroids)) for centroids in all_centroids], run_time=1.5)
        self.wait(1)

        # Run K-means on all 4 grids simultaneously
        for iteration in range(MAX_ITERS):
            all_assignments = []
            all_converged = True

            # Calculate assignments for all grids
            for grid_idx in range(4):
                assignments = []
                for i, (x, y) in enumerate(cities_data):
                    p = all_axes[grid_idx].c2p(x, y)
                    distances = [dist(p, c.get_center()) for c in all_centroids[grid_idx]]
                    closest_idx = distances.index(min(distances))
                    assignments.append(["red", "green", "yellow"][closest_idx])
                all_assignments.append(assignments)

            # Color all dots simultaneously
            color_animations = []
            for grid_idx in range(4):
                for i, assignment in enumerate(all_assignments[grid_idx]):
                    color_animations.append(all_dots[grid_idx][i].animate.set_color(color_for(assignment)))

            self.play(*color_animations, run_time=0.8)
            self.wait(0.3)

            # Calculate new centroids for all grids
            move_animations = []
            reset_animations = []

            for grid_idx in range(4):
                cluster_points = {"red": [], "green": [], "yellow": []}
                for i, assignment in enumerate(all_assignments[grid_idx]):
                    x, y = cities_data[i]
                    cluster_points[assignment].append(all_axes[grid_idx].c2p(x, y))

                new_positions = []
                for cluster_name in ["red", "green", "yellow"]:
                    if len(cluster_points[cluster_name]) > 0:
                        mean_pos = np.mean(np.vstack(cluster_points[cluster_name]), axis=0)
                        mean_pos[2] = 0
                        new_positions.append(mean_pos)
                    else:
                        current_idx = ["red", "green", "yellow"].index(cluster_name)
                        new_positions.append(all_centroids[grid_idx][current_idx].get_center())

                # Check convergence for this grid
                old_positions = [c.get_center() for c in all_centroids[grid_idx]]
                grid_converged = all(dist(old_pos, new_pos) < TOL
                                     for old_pos, new_pos in zip(old_positions, new_positions))

                if not grid_converged:
                    all_converged = False

                # Move centroids
                for i, new_pos in enumerate(new_positions):
                    move_animations.append(all_centroids[grid_idx][i].animate.move_to(new_pos))
                    move_animations.append(all_centroid_Xs[grid_idx][i].animate.move_to(new_pos))

                # Reset dots to grey
                for dot in all_dots[grid_idx]:
                    reset_animations.append(dot.animate.set_color(GREY))

            if all_converged:
                break

            self.play(*move_animations, *reset_animations, run_time=1.2)
            self.wait(0.3)

        # Final coloring for all grids
        final_color_animations = []
        for grid_idx in range(4):
            for i, assignment in enumerate(all_assignments[grid_idx]):
                final_color_animations.append(all_dots[grid_idx][i].animate.set_color(color_for(assignment)))

        self.play(*final_color_animations, run_time=1.0)

        # Remove all centroids
        self.play(*[FadeOut(Group(*centroids)) for centroids in all_centroids], run_time=0.8)

        # Add convex hulls for all grids
        all_shapes = []
        shape_animations = []

        for grid_idx in range(4):
            for cluster_id, cluster_color in enumerate([DOT_RED, DOT_GREEN, DOT_YELLOW]):
                cluster_points = []
                cluster_name = ["red", "green", "yellow"][cluster_id]

                for i, assignment in enumerate(all_assignments[grid_idx]):
                    if assignment == cluster_name:
                        x, y = cities_data[i]
                        screen_point = all_axes[grid_idx].c2p(x, y)
                        cluster_points.append(screen_point[:2])

                if len(cluster_points) >= 3:
                    points_array = np.array(cluster_points)
                    hull = ConvexHull(points_array)
                    vertices = points_array[hull.vertices]
                    vertices_3d = [np.array([x, y, 0]) for x, y in vertices]

                    shape = Polygon(*vertices_3d,
                                    fill_color=cluster_color, fill_opacity=0.25,
                                    stroke_width=0).scale(1.15).set_z_index(-1)
                    all_shapes.append(shape)
                    shape_animations.append(FadeIn(shape))

        # Show all convex hulls simultaneously
        if shape_animations:
            self.play(*shape_animations, run_time=1.5)

        self.wait(1)

        # Calculate and display loss function values - MOVED UP MORE
        loss_texts = []
        loss_animations = []

        for grid_idx in range(4):
            # Calculate final loss (WCSS - Within-Cluster Sum of Squares)
            final_loss = calculate_loss(all_assignments[grid_idx], all_centroids[grid_idx],
                                        cities_data, all_axes[grid_idx])

            # Create loss text below each graph - MOVED UP from 1.1 to 0.9
            loss_text = Text(
                f"Loss: {final_loss:.1f}",
                font_size=24,
                color=WHITE
            ).move_to(positions[grid_idx] + DOWN * 0.5 + RIGHT * 1.98).scale(1.66)  # Changed from 1.1 to 0.9

            loss_texts.append(loss_text)
            loss_animations.append(FadeIn(loss_text))

        # Show all loss values simultaneously
        self.play(*loss_animations, run_time=1.5)

        self.wait(2)

        sq = Rectangle(height=2.96, width=4.5, fill_color="#00ff00", fill_opacity=0.055, stroke_width=0.00000000001)
        sq.shift(LEFT + DOWN).scale(1.24)
        self.play(FadeIn(sq))

        self.wait(2)

        self.play(FadeOut(sq))


