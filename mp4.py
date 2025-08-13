import argparse, os, pickle, numpy as np, cv2
from PIL import Image, ImageDraw
from ocatari.core import OCAtari

def get_rgb(env):
    if hasattr(env, "get_rgb_state") and callable(getattr(env, "get_rgb_state")):
        f = env.get_rgb_state()
        if f is not None: return f
    if hasattr(env, "get_rgb_state"):
        f = env.get_rgb_state
        if f is not None: return f
    return env.render()

def history_to_video(pkl_path, out_path, fps=30):
    with open(pkl_path,'rb') as f:
        d=pickle.load(f)
    states = d["states"]           
    rewards = d.get("rewards", [0]*len(states))

    env = OCAtari(env_name="Skiing", mode="vision", obs_mode="ori",
                  render_mode="rgb_array", buffer_window_size=1)
    env.reset()

    NOOP = 0
    for j, v in enumerate(states[0]): env.set_ram(j, int(v))
    env.step(NOOP)
    frame = get_rgb(env)
    h, w, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for i, ram in enumerate(states):
        for j, v in enumerate(ram): env.set_ram(j, int(v))
        env.step(NOOP)
        rgb = get_rgb(env)
        img = Image.fromarray(rgb, "RGB")

        txt = f"Reward: {int(rewards[i])}" if i < len(rewards) else "Reward: N/A"
        ImageDraw.Draw(img).text((6, h-18), txt, fill=(255,255,255))

        out.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        if i % 1000 == 0:
            print(f"{i}/{len(states)} frames")

    out.release()
    env.close()
    print("Saved:", os.path.abspath(out_path))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="historyXXX_scoreYYY.pkl 경로")
    ap.add_argument("--out", default="replay.mp4", help="저장할 mp4 파일명")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    history_to_video(args.pkl, args.out, fps=args.fps)

#python -u "/Users/koeseoyeon/Desktop/Skiing/mp4.py" \
# --pkl "/Users/koeseoyeon/Desktop/Skiing/##/history1_score-13634.0.pkl" \
# --out "/Users/koeseoyeon/Desktop/Skiing/videos/history1_score-13634.mp4" \
# --fps 30