#!/usr/bin/env python3
"""
PushT Physics Server — matches Diffusion Policy's PushT env EXACTLY.

Reference: github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/pusht/pusht_env.py
"""

import json
import math
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

import pymunk
from pymunk import Vec2d

# ── PushT Environment (matching Diffusion Policy exactly) ────────────

WORKSPACE = 512
SIM_HZ = 100
CONTROL_HZ = 10
N_SUBSTEPS = SIM_HZ // CONTROL_HZ  # 10
DT = 1.0 / SIM_HZ  # 0.01
K_P = 100.0
K_V = 20.0
AGENT_RADIUS = 15.0
TEE_SCALE = 30
TEE_LENGTH = 4


class PushTEnv:
    def __init__(self):
        self.space = None
        self.agent = None
        self.block = None
        self.target_pos = (256.0, 200.0)
        self.target_angle = 0.0

    def reset(self, scenario="push_up"):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0  # matches reference exactly

        # Walls — reference uses (5,5) to (506,506)
        walls = [
            [(5, 506), (5, 5)],
            [(5, 5), (506, 5)],
            [(506, 5), (506, 506)],
            [(5, 506), (506, 506)],
        ]
        for a, b in walls:
            seg = pymunk.Segment(self.space.static_body, a, b, 2)
            seg.elasticity = 0.0
            seg.friction = 1.0
            self.space.add(seg)

        # Agent — KINEMATIC body (reference: add_circle with KINEMATIC)
        self.agent = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.agent.friction = 1  # reference sets friction on body
        agent_shape = pymunk.Circle(self.agent, AGENT_RADIUS)
        agent_shape.filter = pymunk.ShapeFilter()
        self.space.add(self.agent, agent_shape)

        # T-block — reference: add_tee
        self.block = self._add_tee((256, 256), 0)

        # Set positions based on scenario
        if scenario == "push_up":
            self.agent.position = (256, 400)
            self.block.position = (256, 250)
            self.block.angle = 0
            self.target_pos = (256, 100)
            self.target_angle = 0
        elif scenario == "push_right":
            self.agent.position = (100, 271)
            self.block.position = (220, 256)
            self.block.angle = 0
            self.target_pos = (420, 256)
            self.target_angle = 0
        elif scenario == "rotate":
            self.agent.position = (350, 220)
            self.block.position = (256, 256)
            self.block.angle = 0
            self.target_pos = (256, 256)
            self.target_angle = math.pi / 4
        else:
            self.agent.position = (256, 400)
            self.block.position = (256, 250)
            self.block.angle = 0
            self.target_pos = (256, 100)
            self.target_angle = 0

        self.agent.velocity = (0, 0)
        self.block.velocity = (0, 0)
        self.block.angular_velocity = 0

        return self._get_state()

    def _add_tee(self, position, angle):
        """Matches reference add_tee exactly."""
        mass = 1
        scale = TEE_SCALE
        length = TEE_LENGTH

        # Bar vertices (reference order)
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        # Stem vertices
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]

        # Reference uses vertices1 for BOTH moment calculations (likely a bug, but we match it)
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)

        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)

        # Reference sets center_of_gravity to average of both shapes
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2

        body.position = position
        body.angle = angle
        body.friction = 1  # reference sets friction on body

        self.space.add(body, shape1, shape2)
        return body

    def step(self, action, absolute=True):
        """Step with PD control — matches reference EXACTLY.

        Reference:
            acceleration = k_p * (act - pos) + k_v * (Vec2d(0,0) - vel)
            agent.velocity += acceleration * dt

        Args:
            action: [x, y] target position (absolute=True) or [dx, dy] delta (absolute=False)
            absolute: if True, action is absolute position; if False, it's delta
        """
        if absolute:
            target = Vec2d(action[0], action[1])
        else:
            target = Vec2d(
                self.agent.position.x + action[0],
                self.agent.position.y + action[1],
            )

        # Clamp to workspace
        target = Vec2d(
            max(AGENT_RADIUS + 5, min(WORKSPACE - AGENT_RADIUS - 5, target.x)),
            max(AGENT_RADIUS + 5, min(WORKSPACE - AGENT_RADIUS - 5, target.y)),
        )

        for _ in range(N_SUBSTEPS):
            # PD control (reference code, verbatim)
            acceleration = K_P * (target - self.agent.position) + K_V * (
                Vec2d(0, 0) - self.agent.velocity
            )
            self.agent.velocity += acceleration * DT
            self.space.step(DT)

        return self._get_state()

    def reset_with_seed(self, seed):
        """Reset with random initial state matching the original data generation."""
        import random as _random
        rs = _random.Random(seed)

        self.reset("push_up")  # sets up space, walls, shapes

        # Match reference: random positions within bounds
        agent_x = rs.randint(50, 450)
        agent_y = rs.randint(50, 450)
        block_x = rs.randint(100, 400)
        block_y = rs.randint(100, 400)
        block_angle = rs.uniform(-math.pi, math.pi)

        self.agent.position = (agent_x, agent_y)
        self.agent.velocity = (0, 0)
        self.block.position = (block_x, block_y)
        self.block.angle = block_angle
        self.block.velocity = (0, 0)
        self.block.angular_velocity = 0

        # Target T at fixed position (reference default)
        self.target_pos = (256.0, 256.0)
        self.target_angle = 0.0

        return self._get_state()

    def _get_state(self):
        return {
            "agent": {
                "x": round(float(self.agent.position.x), 2),
                "y": round(float(self.agent.position.y), 2),
            },
            "block": {
                "x": round(float(self.block.position.x), 2),
                "y": round(float(self.block.position.y), 2),
                "angle": round(float(self.block.angle), 4),
            },
            "target": {
                "x": round(self.target_pos[0], 2),
                "y": round(self.target_pos[1], 2),
                "angle": round(self.target_angle, 4),
            },
        }


# ── Action Sequences ─────────────────────────────────────────────────


def generate_actions(scenario, steps=60):
    actions = []
    if scenario == "push_up":
        for i in range(steps):
            actions.append([0, -8])
    elif scenario == "push_right":
        for i in range(steps):
            actions.append([8, 0])
    elif scenario == "rotate":
        for i in range(steps):
            if i < 45:
                actions.append([-6, 5])
            else:
                actions.append([-2, 2])
    return actions[:steps]


# ── HTTP Server ──────────────────────────────────────────────────────

env = PushTEnv()


class PushTHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            directory=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            **kwargs,
        )

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""

        if self.path == "/reset":
            data = json.loads(body) if body else {}
            scenario = data.get("scenario", "push_up")
            state = env.reset(scenario)
            self._json_response(state)

        elif self.path == "/step":
            data = json.loads(body)
            action = data.get("action", [0, 0])
            state = env.step(action)
            self._json_response(state)

        elif self.path == "/record":
            data = json.loads(body) if body else {}
            scenario = data.get("scenario", "push_up")
            steps = data.get("steps", 60)
            state = env.reset(scenario)
            trajectory = [state]
            actions = generate_actions(scenario, steps)
            for action in actions:
                state = env.step(action, absolute=False)
                trajectory.append(state)
            self._json_response(
                {"trajectory": trajectory, "actions": actions, "scenario": scenario}
            )

        elif self.path == "/replay_demo":
            data = json.loads(body)
            actions = data.get("actions", [])
            seed = data.get("seed", 0)

            state = env.reset_with_seed(seed)
            trajectory = [state]
            for action in actions:
                state = env.step(action, absolute=True)
                trajectory.append(state)
            self._json_response(
                {"trajectory": trajectory, "actions": actions, "seed": seed}
            )

        else:
            self.send_error(404)

    def _json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        if "POST" in str(args):
            super().log_message(format, *args)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = HTTPServer(("", port), PushTHandler)
    print(f"PushT server on http://localhost:{port}")
    print(f"  POST /reset    — reset environment")
    print(f"  POST /step     — step with action [dx, dy]")
    print(f"  POST /record   — record full trajectory")
    print(f"Serving from: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
