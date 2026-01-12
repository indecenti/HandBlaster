

**Hand-tracking lightgun shooter**. Use MediaPipe hand detection and webcam to control a virtual pistol, shoot moving targets, and survive 7 intense levels.[file:2]

## 🎮 Features

- **Pistol Gesture**: Index finger extended as barrel, thumb free, middle finger curled (smooth temporal filtering)
- **7 Progressive Levels**: Tutorial → Survival → Final Boss with unique target patterns (straight, zigzag, bombs, shields)
- **Weapon System**: Pistol (infinite), Machine Gun (auto-fire), Shotgun (spread)
- **Power-ups**: Health hearts, weapon pickups with hexagonal glow effects
- **Microphone Shooting**: Voice-activated firing (optional calibration)
- **Advanced Effects**: Particle explosions, screen shake, damage popups, trails
- **Calibration System**: 4-point homography mapping with auto-save/load
- **Performance**: 60FPS target, adaptive preview, low-latency tracking

## 📱 Controls

| Input | Action |
|-------|--------|
| Hand Pose | Pistol aiming (index finger → muzzle) |
| SPACE / Voice | Fire weapon |
| Mouse | Menu navigation |
| ESC | Pause/Back |
| 1,2,3 | Quick menu (Mic Calib, Hand Calib, Play) |
| R | Reset calibration |

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/point-black-ultimate.git
cd point-black-ultimate

# Install dependencies
pip install pygame opencv-python mediapipe numpy pyaudio

# Run (webcam + mic required)
python main.py
