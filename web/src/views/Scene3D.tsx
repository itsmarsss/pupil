import { Line, OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import { C } from "../colors";
import { engine, useEngine } from "../engine";
import type { Vec3 } from "../protocol";

// Camera space in cm, the same right-handed y-up convention three.js uses: camera at the origin looking
// toward the user (-z), screen on z = 0 below it.

const POINTS = 478;
const BEZEL_CM = 0.8;
const DECK_CM = 21;

function Lid() {
  const { scene, frame } = useEngine();
  if (!scene) return null;
  const c = scene.screen.corners_cm;
  const screen = [...c, c[0]] as Vec3[];
  const x0 = c[1][0] - BEZEL_CM, x1 = c[0][0] + BEZEL_CM;
  const top = c[0][1] + BEZEL_CM, hinge = c[2][1] - BEZEL_CM * 2;
  const lid: Vec3[] = [[x0, top, 0], [x1, top, 0], [x1, hinge, 0], [x0, hinge, 0], [x0, top, 0]];
  const angle = ((frame?.lid_deg ?? scene.lid_deg ?? 110) * Math.PI) / 180;
  // the deck leaves the hinge at the lid angle, measured from the lid's +y direction toward the user (-z)
  const d: Vec3 = [0, Math.cos(angle) * DECK_CM, -Math.sin(angle) * DECK_CM];
  const deck: Vec3[] = [[x0, hinge, 0], [x1, hinge, 0], [x1, hinge + d[1], d[2]], [x0, hinge + d[1], d[2]], [x0, hinge, 0]];
  return (
    <group>
      <Line points={screen} color={C.screen} lineWidth={2} />
      <Line points={lid} color={C.lid} lineWidth={1} />
      <Line points={deck} color={C.lid} lineWidth={1} />
    </group>
  );
}

function CameraFrustum() {
  const { scene } = useEngine();
  const hfov = ((scene?.camera.hfov_deg ?? 67) * Math.PI) / 180;
  const z = -6, x = Math.tan(hfov / 2) * 6, y = x * (9 / 16);
  const tips: Vec3[] = [[x, y, z], [-x, y, z], [-x, -y, z], [x, -y, z]];
  return (
    <group>
      {tips.map((t, i) => <Line key={i} points={[[0, 0, 0], t]} color="#888" lineWidth={1} />)}
      <Line points={[...tips, tips[0]]} color="#888" lineWidth={1} />
    </group>
  );
}

/** Everything that moves every frame, updated imperatively from the store. */
function Live() {
  const cloud = useRef<THREE.Points>(null!);
  const balls = [useRef<THREE.Mesh>(null!), useRef<THREE.Mesh>(null!)];
  const irises = [useRef<THREE.Mesh>(null!), useRef<THREE.Mesh>(null!)];
  const hits = { ml: useRef<THREE.Mesh>(null!), eyeball: useRef<THREE.Mesh>(null!), fused: useRef<THREE.Mesh>(null!) };
  const target = useRef<THREE.Mesh>(null!);
  const rays = useRef<THREE.LineSegments>(null!);
  const axes = useRef<THREE.LineSegments>(null!);
  const positions = useMemo(() => new Float32Array(POINTS * 3), []);
  const rayPos = useMemo(() => new Float32Array(3 * 2 * 3), []); // two eye rays + face pointing
  const axisPos = useMemo(() => new Float32Array(3 * 2 * 3), []);
  const axisColors = useMemo(() => new Float32Array([1, .3, .3, 1, .3, .3, .3, 1, .3, .3, 1, .3, .3, .5, 1, .3, .5, 1]), []);

  useFrame(() => {
    const f = engine.frame, s = f?.scene;
    const visible = !!s?.points;
    cloud.current.visible = visible;
    if (s?.points) {
      s.points.forEach((p, i) => positions.set(p, i * 3));
      cloud.current.geometry.attributes.position.needsUpdate = true;
    }
    for (let i = 0; i < 2; i++) {
      const c = s?.eyeballs?.[i], ir = s?.iris?.[i];
      balls[i].current.visible = !!c;
      irises[i].current.visible = !!ir;
      if (c) balls[i].current.position.set(...c);
      if (ir) irises[i].current.position.set(...ir);
    }
    // eye rays to their screen hits (or 60 cm along the ray), and the face-pointing ray
    rayPos.fill(0);
    if (s?.eyeballs && s.rays) {
      for (let i = 0; i < 2; i++) {
        const o = s.eyeballs[i], d = s.rays.dirs[i];
        const end = s.rays.hits?.[i] ?? [o[0] + d[0] * 60, o[1] + d[1] * 60, o[2] + d[2] * 60];
        rayPos.set([...o, ...end], i * 6);
      }
    }
    if (s?.head && s.face_pointing) rayPos.set([...s.head.nose, ...s.face_pointing], 12);
    rays.current.geometry.attributes.position.needsUpdate = true;
    rays.current.visible = !!s?.rays || !!s?.face_pointing;

    axisPos.fill(0);
    if (s?.head) {
      const o = s.head.origin;
      s.head.axes.forEach((a, i) => axisPos.set([...o, o[0] + a[0] * 6, o[1] + a[1] * 6, o[2] + a[2] * 6], i * 6));
    }
    axes.current.geometry.attributes.position.needsUpdate = true;
    axes.current.visible = !!s?.head;

    for (const k of ["ml", "eyeball", "fused"] as const) {
      const p = s?.hits?.[k];
      hits[k].current.visible = !!p;
      if (p) hits[k].current.position.set(...p);
    }
    const t = f?.target, sc = engine.scene;
    target.current.visible = !!t && !!sc;
    if (t && sc) {
      // screen points → cm on the lid plane, via the corners (top-left, top-right, bottom-right)
      const [tl, tr, br] = sc.screen.corners_cm;
      const u = t[0] / sc.screen.width_pt, v = t[1] / sc.screen.height_pt;
      target.current.position.set(tl[0] + (tr[0] - tl[0]) * u, tl[1] + (br[1] - tr[1]) * v, 0);
    }
  });

  return (
    <group>
      <points ref={cloud}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        </bufferGeometry>
        <pointsMaterial color={C.face} size={0.18} sizeAttenuation />
      </points>
      {balls.map((r, i) => (
        <mesh key={i} ref={r}>
          <sphereGeometry args={[1.2, 16, 12]} />
          <meshBasicMaterial color="#cfd8dc" wireframe transparent opacity={0.35} />
        </mesh>
      ))}
      {irises.map((r, i) => (
        <mesh key={i} ref={r}>
          <sphereGeometry args={[0.28, 12, 8]} />
          <meshBasicMaterial color={C.iris} />
        </mesh>
      ))}
      <lineSegments ref={rays}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[rayPos, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color={C.rays} />
      </lineSegments>
      <lineSegments ref={axes}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[axisPos, 3]} />
          <bufferAttribute attach="attributes-color" args={[axisColors, 3]} />
        </bufferGeometry>
        <lineBasicMaterial vertexColors />
      </lineSegments>
      {(["ml", "eyeball", "fused"] as const).map((k) => (
        <mesh key={k} ref={hits[k]}>
          <sphereGeometry args={[k === "fused" ? 0.7 : 0.5, 16, 12]} />
          <meshBasicMaterial color={C[k]} />
        </mesh>
      ))}
      <mesh ref={target}>
        <ringGeometry args={[0.6, 0.9, 24]} />
        <meshBasicMaterial color={C.target} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

/** Keep the orbit centre between the screen and the head, so the face stays in view as it moves. */
function FollowHead() {
  const controls = useThree((st) => st.controls) as unknown as { target: THREE.Vector3; update: () => void } | null;
  const goal = useMemo(() => new THREE.Vector3(0, -6, -25), []);
  useFrame(() => {
    const o = engine.frame?.scene?.head?.origin;
    if (!controls || !o) return;
    goal.set(o[0] / 2, (o[1] - 10) / 2, o[2] / 2);
    controls.target.lerp(goal, 0.05);
    controls.update();
  });
  return null;
}

export function Scene3D() {
  return (
    <Canvas camera={{ position: [55, 25, -75], fov: 45, near: 0.5, far: 500 }}>
      <color attach="background" args={["#0f1115"]} />
      <OrbitControls target={[0, -6, -25]} makeDefault />
      <FollowHead />
      <CameraFrustum />
      <Lid />
      <Live />
    </Canvas>
  );
}
