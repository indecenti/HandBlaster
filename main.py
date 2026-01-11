import pygame
import cv2
import mediapipe as mp
import numpy as np
import time
import collections
import pickle
import os
import random
import math
import pyaudio
import struct
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

print("=" * 80)
print("  ██████╗  ██████╗ ██╗███╗   ██╗████████╗    ██████╗ ██╗      █████╗  ██████╗██╗  ██╗")
print("  ██╔══██╗██╔═══██╗██║████╗  ██║╚══██╔══╝    ██╔══██╗██║     ██╔══██╗██╔════╝██║ ██╔╝")
print("  ██████╔╝██║   ██║██║██╔██╗ ██║   ██║       ██████╔╝██║     ███████║██║     █████╔╝ ")
print("  ██╔═══╝ ██║   ██║██║██║╚██╗██║   ██║       ██╔══██╗██║     ██╔══██║██║     ██╔═██╗ ")
print("  ██║     ╚██████╔╝██║██║ ╚████║   ██║       ██████╔╝███████╗██║  ██║╚██████╗██║  ██╗")
print("  ╚═╝      ╚═════╝ ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝")
print("=" * 80)
print("   ULTIMATE v8.7 - POLISHED | CALIBRATION FIX | VISUAL IMPROVEMENTS")
print("=" * 80)

# ==================== CONSTANTS ====================
SCREEN_WIDTH: int = 1280
SCREEN_HEIGHT: int = 720
TARGET_FPS: int = 60

# Colors
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (0, 255, 255)
COLOR_RED = (255, 50, 50)
COLOR_GREEN = (100, 255, 100)
COLOR_YELLOW = (255, 255, 100)
COLOR_ORANGE = (255, 150, 50)
COLOR_GOLD = (255, 215, 0)
COLOR_NEON_GREEN = (57, 255, 20)  # Neon green for calibration
COLOR_NEON_PINK = (255, 16, 240)  # Neon pink for active points

# ==================== PYGAME INIT ====================
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Point Black ULTIMATE v8.7")
clock = pygame.time.Clock()

# Fonts
font_title = pygame.font.Font(None, 72)
font_subtitle = pygame.font.Font(None, 48)
font_large = pygame.font.Font(None, 42)
font = pygame.font.Font(None, 40)
font_small = pygame.font.Font(None, 28)
font_tiny = pygame.font.Font(None, 22)

# ==================== AUDIO SYSTEM ====================
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

audio = None
stream = None
mic_available = False
mic_baseline = 100
mic_threshold = 1500
mic_calibrated = False

try:
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    print("🎤 Microphone initialized!")
    mic_available = True
except Exception as e:
    print(f"⚠️ Microphone not available: {e}")
    mic_available = False
    stream = None

def get_audio_level() -> int:
    if not mic_available or stream is None:
        return 0
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        count = len(data) / 2
        format_str = "%dh" % count
        shorts = struct.unpack(format_str, data)
        sum_squares = sum([sample ** 2 for sample in shorts])
        rms = math.sqrt(sum_squares / count)
        return int(rms)
    except:
        return 0

# ==================== WEBCAM SETUP ====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

WEB_CAM_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
WEB_CAM_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"🎥 Webcam: {WEB_CAM_WIDTH}x{WEB_CAM_HEIGHT}")

# Smaller, less invasive preview
PREVIEW_W, PREVIEW_H = 180, int(180 * WEB_CAM_HEIGHT / max(WEB_CAM_WIDTH, 1))
preview_rect = pygame.Rect(SCREEN_WIDTH - PREVIEW_W - 10, SCREEN_HEIGHT - PREVIEW_H - 10, PREVIEW_W, PREVIEW_H)

# ==================== MEDIAPIPE ====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.4,
    model_complexity=0
)

# ==================== CALIBRATION ====================
MARGIN = 120
screen_pts = np.array([
    [MARGIN, MARGIN],
    [SCREEN_WIDTH - MARGIN, MARGIN],
    [SCREEN_WIDTH - MARGIN, SCREEN_HEIGHT - MARGIN],
    [MARGIN, SCREEN_HEIGHT - MARGIN]
], dtype=np.float32)

calib_labels = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT"]
CALIB_FILE = "pointblack_ultimate_v87.pkl"

# ==================== WEAPON SYSTEM ====================
@dataclass
class WeaponStats:
    name: str
    fire_rate: int
    damage: int
    ammo: int
    max_ammo: int
    auto_fire: bool
    spread: int
    spread_angle: float
    color: Tuple[int, int, int]
    tracer_width: int

WEAPONS = {
    'PISTOL': WeaponStats(
        name='PISTOL',
        fire_rate=12,
        damage=1,
        ammo=-1,
        max_ammo=-1,
        auto_fire=False,
        spread=1,
        spread_angle=0,
        color=(100, 230, 255),
        tracer_width=4
    ),
    'MACHINEGUN': WeaponStats(
        name='MACHINE GUN',
        fire_rate=3,
        damage=1,
        ammo=60,
        max_ammo=60,
        auto_fire=True,
        spread=1,
        spread_angle=0,
        color=(255, 200, 0),
        tracer_width=3
    ),
    'SHOTGUN': WeaponStats(
        name='SHOTGUN',
        fire_rate=25,
        damage=2,
        ammo=8,
        max_ammo=8,
        auto_fire=False,
        spread=9,
        spread_angle=25,
        color=(255, 100, 50),
        tracer_width=5
    )
}

# ==================== POWER-UP SYSTEM ====================
@dataclass
class PowerUpType:
    name: str
    icon: str
    color: Tuple[int, int, int]
    glow: Tuple[int, int, int]
    effect_type: str
    weapon_type: Optional[str] = None
    value: int = 0

POWERUP_TYPES = {
    'HEALTH': PowerUpType(
        name='HEALTH',
        icon='♥',
        color=(255, 20, 147),      # neon pink
        glow=(255, 100, 200),
        effect_type='health',
        value=1
    ),
    'MACHINEGUN': PowerUpType(
        name='MACHINE GUN',
        icon='⚡',
        color=(255, 200, 0),
        glow=(200, 150, 0),
        effect_type='weapon',
        weapon_type='MACHINEGUN'
    ),
    'SHOTGUN': PowerUpType(
        name='SHOTGUN',
        icon='✦',
        color=(255, 100, 50),
        glow=(200, 50, 0),
        effect_type='weapon',
        weapon_type='SHOTGUN'
    )
}

# ==================== LEVEL SYSTEM ====================
LEVELS = [
    {'id': 1, 'name': 'TUTORIAL', 'targets_to_spawn': 20, 'time_limit': 60, 'spawn_rate': 50, 'max_targets': 3,
     'target_types': ['EASY', 'MEDIUM'], 'powerup_chance': 0.20, 'background_color': (10, 20, 40)},
    {'id': 2, 'name': 'WARMING UP', 'targets_to_spawn': 30, 'time_limit': 50, 'spawn_rate': 40, 'max_targets': 4,
     'target_types': ['EASY', 'MEDIUM', 'HARD'], 'powerup_chance': 0.22, 'background_color': (15, 25, 45)},
    {'id': 3, 'name': 'SPEED TRIAL', 'targets_to_spawn': 35, 'time_limit': 45, 'spawn_rate': 30, 'max_targets': 5,
     'target_types': ['MEDIUM', 'HARD', 'SPEED'], 'powerup_chance': 0.25, 'background_color': (20, 15, 50)},
    {'id': 4, 'name': 'CHAOS MODE', 'targets_to_spawn': 40, 'time_limit': 50, 'spawn_rate': 25, 'max_targets': 6,
     'target_types': ['HARD', 'SPEED', 'ZIGZAG', 'BONUS'], 'powerup_chance': 0.28, 'background_color': (30, 10, 40)},
    {'id': 5, 'name': 'NIGHTMARE', 'targets_to_spawn': 45, 'time_limit': 45, 'spawn_rate': 20, 'max_targets': 7,
     'target_types': ['HARD', 'SPEED', 'ZIGZAG', 'SHIELD', 'BOMB'], 'powerup_chance': 0.30, 'background_color': (40, 5, 30)},
    {'id': 6, 'name': 'SURVIVAL', 'targets_to_spawn': 55, 'time_limit': 40, 'spawn_rate': 15, 'max_targets': 8,
     'target_types': ['SPEED', 'ZIGZAG', 'SHIELD', 'BOMB', 'BONUS'], 'powerup_chance': 0.35, 'background_color': (50, 10, 10)},
    {'id': 7, 'name': 'FINAL BOSS', 'targets_to_spawn': 1, 'time_limit': 90, 'spawn_rate': 999, 'max_targets': 1,
     'target_types': ['BOSS'], 'powerup_chance': 0.0, 'background_color': (10, 5, 50)}
]

# ==================== TARGET TYPES ====================
TARGET_TYPES = {
    'EASY': {'name': 'Easy', 'min_r': 60, 'max_r': 75, 'speed_min': 1.5, 'speed_max': 2.5,
             'points': 10, 'hp': 1, 'color': (100, 255, 100), 'glow': (50, 200, 50), 'pattern': 'straight'},
    'MEDIUM': {'name': 'Medium', 'min_r': 45, 'max_r': 55, 'speed_min': 2.5, 'speed_max': 3.8,
               'points': 25, 'hp': 1, 'color': (255, 200, 50), 'glow': (255, 150, 0), 'pattern': 'straight'},
    'HARD': {'name': 'Hard', 'min_r': 32, 'max_r': 42, 'speed_min': 3.5, 'speed_max': 5.0,
             'points': 50, 'hp': 1, 'color': (255, 50, 50), 'glow': (200, 0, 0), 'pattern': 'straight'},
    'SPEED': {'name': 'Speed', 'min_r': 25, 'max_r': 35, 'speed_min': 5.0, 'speed_max': 7.5,
              'points': 75, 'hp': 1, 'color': (0, 255, 255), 'glow': (0, 150, 200), 'pattern': 'straight'},
    'ZIGZAG': {'name': 'ZigZag', 'min_r': 38, 'max_r': 48, 'speed_min': 2.8, 'speed_max': 4.2,
               'points': 60, 'hp': 1, 'color': (255, 0, 255), 'glow': (200, 0, 200), 'pattern': 'zigzag'},
    'BONUS': {'name': 'Bonus', 'min_r': 50, 'max_r': 65, 'speed_min': 2.0, 'speed_max': 3.5,
              'points': 100, 'hp': 1, 'color': (255, 215, 0), 'glow': (255, 180, 0), 'pattern': 'straight', 'bonus_life': True},
    'BOMB': {'name': 'Bomb', 'min_r': 40, 'max_r': 50, 'speed_min': 2.2, 'speed_max': 3.8,
             'points': -50, 'hp': 1, 'color': (50, 50, 50), 'glow': (100, 0, 0), 'pattern': 'straight', 'penalty': True},
    'SHIELD': {'name': 'Shield', 'min_r': 48, 'max_r': 58, 'speed_min': 2.5, 'speed_max': 3.5,
               'points': 80, 'hp': 2, 'color': (100, 150, 255), 'glow': (50, 100, 200), 'pattern': 'straight'},
    'BOSS': {'name': 'BOSS', 'min_r': 120, 'max_r': 120, 'speed_min': 1.0, 'speed_max': 1.0,
             'points': 1000, 'hp': 50, 'color': (255, 0, 100), 'glow': (150, 0, 50), 'pattern': 'boss'}
}

# ==================== GAME STATE ====================
state = 'menu'
calib_step = 0
calib_pts = []
homography: Optional[np.ndarray] = None

# Mic calibration state
mic_calib_state = None
mic_calib_start = 0.0
mic_silence_samples = []
mic_noise_samples = []

# Player
pistol_center_raw: Optional[np.ndarray] = None
pistol_grip_raw: Optional[np.ndarray] = None
pistol_confidence: float = 0.0
gun_screen_pos = np.array([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2], dtype=np.float32)
screen_grip_pos = np.array([SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2], dtype=np.float32)

# Weapon system
current_weapon = 'PISTOL'
weapon_ammo = WEAPONS[current_weapon].ammo
fire_cooldown = 0
last_shot_time = 0.0

# Audio shooting
audio_level = 0
last_voice_shot = 0.0
voice_cooldown = 0.15

# Game
current_level: int = 0
score: int = 0
displayed_score: int = 0
combo: int = 0
max_combo: int = 0
lives: int = 5
accuracy_shots: int = 0
accuracy_hits: int = 0
targets: List = []
particles: List = []
damage_numbers: List = []
powerups: List = []
projectiles: List = []

# Level
targets_spawned: int = 0
level_time: float = 0.0
level_start_time: float = 0.0
level_complete_timer: float = 0.0
level_complete_delay: float = 2.0

# Effects
screen_shake: int = 0
flash_alpha: int = 0

# Tracking
TRAIL_LENGTH = 10
muzzle_trail = collections.deque([gun_screen_pos.copy()] * TRAIL_LENGTH, maxlen=TRAIL_LENGTH)
grip_trail = collections.deque([screen_grip_pos.copy()] * TRAIL_LENGTH, maxlen=TRAIL_LENGTH)
gun_velocity = np.array([0.0, 0.0])

# Constants
PISTOL_CONFIDENCE_DECAY = 0.97
MIN_PISTOL_CONFIDENCE = 0.18
VELOCITY_ALPHA = 0.3
FPS_HISTORY = collections.deque(maxlen=60)
FRAME_SKIP_COUNTER = 0
PREVIEW_SKIP = 3
avg_fps = 60.0

# Background
stars: List = []
for _ in range(200):
    stars.append({'x': random.randint(0, SCREEN_WIDTH), 'y': random.randint(0, SCREEN_HEIGHT),
                  'z': random.randint(1, 100), 'speed': random.uniform(0.5, 2.0)})

# ==================== PARTICLE CLASSES ====================
class ExplosionParticle:
    def __init__(self, x: float, y: float, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 12)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.lifetime = random.randint(20, 40)
        self.max_lifetime = self.lifetime
        self.size = random.randint(3, 8)

    def update(self) -> bool:
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.3
        self.vx *= 0.97
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, surface: pygame.Surface):
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        size = max(1, int(self.size * (self.lifetime / self.max_lifetime)))
        if 0 <= self.x < SCREEN_WIDTH and 0 <= self.y < SCREEN_HEIGHT:
            pygame.draw.circle(surface, (*self.color, min(alpha, 255)), (int(self.x), int(self.y)), size)

class ShockwaveParticle:
    def __init__(self, x: float, y: float, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.radius = 5.0
        self.max_radius = random.randint(40, 70)
        self.color = color
        self.lifetime = 20
        self.max_lifetime = self.lifetime

    def update(self) -> bool:
        self.radius += (self.max_radius - self.radius) * 0.25
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, surface: pygame.Surface):
        alpha = int(180 * (self.lifetime / self.max_lifetime))
        if 0 <= self.x < SCREEN_WIDTH and 0 <= self.y < SCREEN_HEIGHT:
            surf = pygame.Surface((int(self.radius*2.5), int(self.radius*2.5)), pygame.SRCALPHA)
            center = (int(self.radius*1.25), int(self.radius*1.25))
            pygame.draw.circle(surf, (*self.color, alpha), center, int(self.radius), 3)
            surface.blit(surf, (int(self.x - self.radius*1.25), int(self.y - self.radius*1.25)))










class DamageNumber:
    def __init__(self, x: float, y: float, damage: int, critical: bool = False):
        self.x = x
        self.y = y
        self.damage = abs(damage)
        self.is_damage = damage < 0
        self.critical = critical
        self.lifetime = 60
        self.max_lifetime = self.lifetime
        
        # FISICA PESANTE (Salto deciso)
        self.vy = -6.5              # Salto più alto
        self.vx = random.uniform(-1.5, 1.5)
        
        # ANIMAZIONE SCALA (Enorme e definita)
        self.scale = 0.5
        self.target_scale = 2.2 if critical else 1.6  # 220% dimensione originale
        self.scale_speed = 0.18
        self.alpha = 255
        
    def update(self) -> bool:
        # Fisica fluida
        self.x += self.vx * 0.96
        self.y += self.vy
        self.vy += 0.3  # Gravità arcade veloce
        
        # Scale Interpolation (No oscillazioni strane)
        self.scale += (self.target_scale - self.scale) * self.scale_speed
        
        # Fade out pulito finale
        if self.lifetime < 15:
            self.target_scale *= 1.05 # Leggera espansione finale mentre svanisce
            self.alpha = int(255 * (self.lifetime / 15))
            
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, surface: pygame.Surface):
        if not (0 <= self.x < SCREEN_WIDTH and 0 <= self.y < SCREEN_HEIGHT):
            return
            
        text = f"{self.damage}"
        if self.critical: text += "!"
        elif self.is_damage: text = f"-{self.damage}"
        
        # COLORI PIATTI E DEFINITI (Zero sfumature sporche)
        if self.critical:
            color = (255, 255, 60)      # Giallo puro
            shadow_color = (160, 40, 0) # Ruggine scuro
        elif self.is_damage:
            color = (255, 60, 60)       # Rosso puro
            shadow_color = (100, 20, 20)
        else:
            color = (60, 255, 100)      # Verde neon
            shadow_color = (20, 80, 40)
            
        # 1. RENDERIZZA TESTO BASE (Usa sempre font grande)
        # Usiamo 'font' (40pt) per massima risoluzione base
        src_font = font 
        
        text_surf = src_font.render(text, True, color)
        shadow_surf = src_font.render(text, True, shadow_color)
        
        # 2. CALCOLA DIMENSIONI (Smoothscale per evitare pixel sgranati)
        target_w = int(text_surf.get_width() * self.scale)
        target_h = int(text_surf.get_height() * self.scale)
        
        if target_w <= 0 or target_h <= 0: return

        # SMOOTHSCALE: Fondamentale per ingrandire senza "scalettature"
        scaled_text = pygame.transform.smoothscale(text_surf, (target_w, target_h))
        scaled_shadow = pygame.transform.smoothscale(shadow_surf, (target_w, target_h))
        
        # 3. GESTIONE ALPHA (Trasparenza globale)
        if self.alpha < 255:
            scaled_text.set_alpha(self.alpha)
            scaled_shadow.set_alpha(self.alpha)
            
        # 4. DISEGNO CON HARD SHADOW (Massima definizione)
        cx, cy = int(self.x), int(self.y)
        
        # L'ombra scala con il testo (più grande il testo, più distante l'ombra)
        shadow_offset = max(2, int(5 * self.scale)) 
        
        # Disegna OMBRA (Spostata in basso a destra)
        surface.blit(scaled_shadow, (cx - target_w//2 + shadow_offset, cy - target_h//2 + shadow_offset))
        
        # Disegna TESTO (Sopra)
        surface.blit(scaled_text, (cx - target_w//2, cy - target_h//2))




class Projectile:
    def __init__(self, x: float, y: float, angle_offset: float, weapon_type: str):
        self.x = x
        self.y = y
        base_angle = -math.pi / 2
        self.angle = base_angle + angle_offset
        self.speed = 30
        self.vx = math.cos(self.angle) * self.speed
        self.vy = math.sin(self.angle) * self.speed
        self.weapon_type = weapon_type
        self.weapon_stats = WEAPONS[weapon_type]
        self.lifetime = 30
        self.hit = False

    def update(self) -> bool:
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        return self.lifetime > 0 and not self.hit and 0 <= self.x < SCREEN_WIDTH and 0 <= self.y < SCREEN_HEIGHT

    def draw(self, surface: pygame.Surface):
        end_x = self.x - self.vx * 0.6
        end_y = self.y - self.vy * 0.6
        pygame.draw.line(surface, self.weapon_stats.color, (int(self.x), int(self.y)), 
                        (int(end_x), int(end_y)), self.weapon_stats.tracer_width)









def create_explosion(x: float, y: float, color: Tuple[int, int, int], intensity: float = 1.0):
    global particles, screen_shake
    count = int(15 * intensity)
    for _ in range(count):
        particles.append(ExplosionParticle(x, y, color))
    particles.append(ShockwaveParticle(x, y, color))
    screen_shake = max(screen_shake, int(6 * intensity))

# ==================== MENU BUTTON CLASS ====================
class MenuButton:
    def __init__(self, x: int, y: int, width: int, height: int, text: str, action: str, enabled: bool = True):
        self.rect = pygame.Rect(x - width // 2, y - height // 2, width, height)
        self.text = text
        self.action = action
        self.enabled = enabled
        self.hovered = False
        self.pulse = 0.0

    def check_hover(self, pos: Tuple[int, int]) -> bool:
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered

    def update(self):
        self.pulse = math.sin(time.time() * 3) * 0.5 + 0.5

    def draw(self, surface: pygame.Surface):
        self.update()
        if not self.enabled:
            color = (80, 80, 80)
            text_color = (150, 150, 150)
        elif self.hovered:
            color = (0, 200, 255)
            text_color = COLOR_WHITE
        else:
            color = (50, 100, 150)
            text_color = (200, 200, 200)

        if self.hovered:
            glow_surf = pygame.Surface((self.rect.width + 30, self.rect.height + 30), pygame.SRCALPHA)
            glow_alpha = int(80 + 60 * self.pulse)
            pygame.draw.rect(glow_surf, (*color, glow_alpha), glow_surf.get_rect(), border_radius=15)
            surface.blit(glow_surf, (self.rect.x - 15, self.rect.y - 15))

        pygame.draw.rect(surface, (*color, 200), self.rect, border_radius=10)
        border_color = COLOR_WHITE if self.hovered else (100, 150, 200)
        pygame.draw.rect(surface, border_color, self.rect, 3 if self.hovered else 2, border_radius=10)

        text_surf = font_large.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

        if self.hovered and self.enabled:
            pygame.draw.circle(surface, (255, 255, 0), (self.rect.left - 20, self.rect.centery), 
                             int(12 + 3 * self.pulse))






# ==================== MENU SETUP ====================
menu_buttons: List[MenuButton] = []

def init_menu_buttons():
    global menu_buttons
    menu_buttons = [
        MenuButton(SCREEN_WIDTH // 2, 320, 400, 65, "MIC CALIBRATION", "mic_calib", mic_available),
        MenuButton(SCREEN_WIDTH // 2, 400, 400, 65, "HAND CALIBRATION", "calib", True),
        MenuButton(SCREEN_WIDTH // 2, 480, 400, 65, "START GAME", "play", homography is not None),
        MenuButton(SCREEN_WIDTH // 2, 560, 400, 65, "RESET CALIB", "reset", True)
    ]

init_menu_buttons()
































# ==================== CORE FUNCTIONS ====================

def enhance_bright(raw_frame: np.ndarray) -> np.ndarray:
    gamma = 0.65
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    gamma_frame = cv2.LUT(raw_frame, table)
    return cv2.convertScaleAbs(gamma_frame, alpha=1.4, beta=30)


def save_calibration() -> bool:
    """Salva homography + calib_pts con DEBUG completo"""
    global homography, calib_pts
    try:
        print(f"💾 SAVING: homography={homography is not None}, calib_pts={len(calib_pts)} points")
        if homography is None or len(calib_pts) != 4:
            print("❌ INVALID DATA - NOT SAVING")
            return False
            
        data = {
            'homography': homography.tolist() if homography is not None else None,
            'calib_pts': [[p[0], p[1]] for p in calib_pts],
            'timestamp': time.time()
        }
        
        with open(CALIB_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ SAVED to {CALIB_FILE} - {len(data['calib_pts'])} points")
        return True
        
    except Exception as e:
        print(f"❌ SAVE ERROR: {e}")
        return False

def load_calibration() -> bool:
    """Carica homography + calib_pts con VALIDAZIONE"""
    global homography, calib_pts
    try:
        if not os.path.exists(CALIB_FILE):
            print(f"❌ NO CALIB FILE: {CALIB_FILE}")
            return False
            
        print(f"📂 LOADING from {CALIB_FILE}")
        with open(CALIB_FILE, 'rb') as f:
            data = pickle.load(f)
        
        # VALIDAZIONE DATI
        if 'homography' not in data or data['homography'] is None:
            print("❌ INVALID homography in file")
            return False
            
        if 'calib_pts' not in data or len(data['calib_pts']) != 4:
            print(f"❌ INVALID calib_pts: {len(data.get('calib_pts', []))}/4")
            return False
        
        # CONVERTE IN NUMPY
        homography = np.array(data['homography'], dtype=np.float32)
        calib_pts = [np.array(p, dtype=np.float32) for p in data['calib_pts']]
        
        print(f"✅ LOADED: homography={homography.shape}, calib_pts={len(calib_pts)} points")
        return True
        
    except Exception as e:
        print(f"❌ LOAD ERROR: {e}")
        return False



load_calibration()


# AGGIUNGI GLOBALI (in cima al file con gli altri globali)
pistol_score_history = collections.deque([0.0] * 10, maxlen=10)
pistol_grace_period = 0  # Frame di grazia dopo perdita tracking
PISTOL_GRACE_FRAMES = 8  # ~133ms @ 60fps mantiene pose dopo movimento veloce

def is_pistol_pose(hand_lm) -> float:
    """
    Score pistola PERSISTENTE con criteri rilassati per gaming:
    - Trigger facile (indice esteso + pollice libero)
    - Pesi bilanciati per massima rilevazione
    - Zero calcoli complessi che causano jitter
    
    Returns: score [0.0, 1.0] con bias verso detection
    """
    def to_np(lm):
        return np.array([lm.x, lm.y])
    
    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)
    
    landmarks = hand_lm.landmark
    
    # Punti essenziali
    wrist = to_np(landmarks[mp_hands.HandLandmark.WRIST])
    thumb_tip = to_np(landmarks[mp_hands.HandLandmark.THUMB_TIP])
    thumb_mcp = to_np(landmarks[mp_hands.HandLandmark.THUMB_MCP])
    index_tip = to_np(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    index_mcp = to_np(landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP])
    middle_tip = to_np(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
    middle_mcp = to_np(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP])
    
    score = 0.0
    
    # 1. INDICE ESTESO (50% peso) - Soglia MOLTO rilassata
    index_dist = dist(index_tip, wrist)
    index_mcp_dist = dist(index_mcp, wrist)
    if index_dist > index_mcp_dist * 1.08:  # 1.12->1.08 più tollerante
        score += 0.50
    else:
        score += max(0.0, (index_dist / (index_mcp_dist * 1.08) - 0.9) * 5.0)  # Decadimento graduale
    
    # 2. POLLICE LIBERO (30% peso) - Separato da palmo
    thumb_dist = dist(thumb_tip, wrist)
    thumb_mcp_dist = dist(thumb_mcp, wrist)
    if thumb_dist > thumb_mcp_dist * 1.05:  # 1.1->1.05 più facile
        score += 0.30
    else:
        score += max(0.0, (thumb_dist / (thumb_mcp_dist * 1.05) - 0.95) * 6.0)
    
    # 3. MEDIO PIEGATO (20% peso) - Almeno 1 dito piegato conferma grip
    middle_dist = dist(middle_tip, wrist)
    middle_mcp_dist_norm = dist(middle_mcp, wrist)
    if middle_dist < middle_mcp_dist_norm * 0.90:  # 0.85->0.90 più tollerante
        score += 0.20
    else:
        score += max(0.0, 0.20 - (middle_dist / middle_mcp_dist_norm - 0.90) * 2.0)
    
    return min(1.0, score)

def detect_realistic_pistol(raw_frame: np.ndarray) -> Tuple:
    """
    Rilevamento pistola ULTRA-PERSISTENTE con:
    - Smooth temporale aggressivo (rolling avg 10 frame)
    - Grace period 8 frame dopo perdita tracking
    - Confidence decay lento (0.95 invece 0.97)
    """
    global pistol_confidence, pistol_grace_period, pistol_center_raw, pistol_grip_raw, pistol_score_history
    global pistol_score_history, pistol_grace_period

    bright_frame = enhance_bright(raw_frame)
    rgb = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    center_raw = None
    grip_raw = None
    muzzleoverlay = None
    gripoverlay = None
    
    if results.multi_hand_landmarks:
        hand_lm = results.multi_hand_landmarks[0]
        pose_score = is_pistol_pose(hand_lm)
        
        # FILTRAGGIO TEMPORALE: media mobile 10 frame
        pistol_score_history.append(pose_score)
        smoothed_score = np.mean(list(pistol_score_history))
        
        # Calcola punti muzzle/grip
        wrist = hand_lm.landmark[mp_hands.HandLandmark.WRIST]
        index_tip = hand_lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        
        wx, wy = int(wrist.x * WEB_CAM_WIDTH), int(wrist.y * WEB_CAM_HEIGHT)
        tipx, tipy = int(index_tip.x * WEB_CAM_WIDTH), int(index_tip.y * WEB_CAM_HEIGHT)
        pipx, pipy = int(index_pip.x * WEB_CAM_WIDTH), int(index_pip.y * WEB_CAM_HEIGHT)
        
        cx = int(pipx * 0.4 + tipx * 0.6)
        cy = int(pipy * 0.4 + tipy * 0.6)
        
        center_raw = np.array([cx, cy], dtype=np.float32)
        grip_raw = np.array([wx, wy], dtype=np.float32)
        muzzleoverlay = (cx, cy)
        gripoverlay = (wx, wy)
        
        # UPDATE CONFIDENCE: blend score smooth con history
        pistol_confidence = 0.85 * smoothed_score + 0.15 * pistol_confidence  # Reattivo ma stabile
        pistol_grace_period = PISTOL_GRACE_FRAMES  # Reset grace period
        
    else:
        # GRACE PERIOD: mantieni confidence durante perdita temporanea
        if pistol_grace_period > 0:
            pistol_grace_period -= 1
            pistol_confidence *= 0.95  # Decay lento (era 0.97 troppo veloce)
        else:
            pistol_confidence *= 0.88  # Decay accelerato dopo grace period
    
    # Memorizza punti raw anche senza tracking (usa ultimo valido)
    pistol_center_raw = center_raw
    pistol_grip_raw = grip_raw
    
    # 🔧 CLEANUP CRITICO - SEMPRE PRIMA DEL RETURN
    results = None
    del results
    
    # SOGLIA BASSA per attivazione: 0.15 (era 0.2)
    if pistol_confidence > 0.15:
        return center_raw, grip_raw, muzzleoverlay, gripoverlay
    
    return None, None, None, None






def detect_realistic_pistol_bkp(raw_frame: np.ndarray) -> Tuple:
    global pistol_confidence, pistol_center_raw, pistol_grip_raw

    bright_frame = enhance_bright(raw_frame)
    rgb = cv2.cvtColor(bright_frame, cv2.COLOR_BGR2RGB)
    # DOPO hands.process(), AGGIUNGI:
    if results.multihandlandmarks:
        # usa results
        results = None  # ← FORCE GC
        del results

    # OPPURE usa with context manager:
    with mp.solutions.hands.Hands(...) as hands:
        results = hands.process(rgb)


    center_raw = None
    grip_raw = None
    muzzle_overlay = None
    grip_overlay = None

    if results.multi_hand_landmarks:
        hand_lm = results.multi_hand_landmarks[0]
        pose_score = is_pistol_pose(hand_lm)

        wrist = hand_lm.landmark[mp_hands.HandLandmark.WRIST]
        index_tip = hand_lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

        wx = int(wrist.x * WEB_CAM_WIDTH)
        wy = int(wrist.y * WEB_CAM_HEIGHT)
        tipx = int(index_tip.x * WEB_CAM_WIDTH)
        tipy = int(index_tip.y * WEB_CAM_HEIGHT)
        pipx = int(index_pip.x * WEB_CAM_WIDTH)
        pipy = int(index_pip.y * WEB_CAM_HEIGHT)

        cx = int(pipx * 0.4 + tipx * 0.6)
        cy = int(pipy * 0.4 + tipy * 0.6)

        center_raw = np.array([cx, cy], dtype=np.float32)
        grip_raw = np.array([wx, wy], dtype=np.float32)
        muzzle_overlay = (cx, cy)
        grip_overlay = (wx, wy)

        pistol_confidence = 0.9 * pose_score + 0.1 * pistol_confidence
    else:
        pistol_confidence *= PISTOL_CONFIDENCE_DECAY

    pistol_center_raw = center_raw
    pistol_grip_raw = grip_raw

    if pistol_confidence > 0.2:
        return center_raw, grip_raw, muzzle_overlay, grip_overlay
    return None, None, None, None












def compute_homography_pro() -> bool:
    """
    HOMOGRAPHY ULTIMATE v9.1 - DATA FORMAT FIXED
    ✅ Auto-detect calib_pts format (list/list/numpy)
    ✅ Robust normalization + Multi-method
    ✅ Zero crash garantie
    """
    global homography, calib_pts
    print(f"🔧 COMPUTING ULTIMATE v9.1 - calib_pts={len(calib_pts)}")
    
    if len(calib_pts) != 4:
        print(f"❌ BAD CALIB_PTS: {len(calib_pts)}/4")
        return False
    
    # ========== DATA NORMALIZATION ==========
    def normalize_calib_pts(pts_list):
        """Converte QUALSIASI formato in np.array (N,2)"""
        if isinstance(pts_list, np.ndarray):
            pts = pts_list
        elif isinstance(pts_list[0], np.ndarray):
            pts = np.array([p.tolist() for p in pts_list])
        else:
            pts = np.array(pts_list)
        
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        elif pts.ndim == 3 and pts.shape[1:] == (1, 2):
            pts = pts.squeeze(1)
            
        print(f"📐 calib_pts normalized: {pts.shape} = {pts}")
        return pts.astype(np.float32)
    
    try:
        # NORMALIZZA FORMAT
        srcpts = normalize_calib_pts(calib_pts).reshape(-1, 1, 2)
        dstpts = normalize_calib_pts(screen_pts).reshape(-1, 1, 2)
        
        print(f"✅ Shapes OK: src={srcpts.shape}, dst={dstpts.shape}")
        
        # ========== MULTI-METHOD SIMPLE ==========
        methods = [
            ("RANSAC", cv2.RANSAC, 3.0),
            ("LMEDS", cv2.LMEDS, 5.0),
            ("DLT", 0, 0)
        ]
        
        best_H = None
        best_reproj = float('inf')
        
        for method_name, method, thresh in methods:
            try:
                if method == 0:
                    H_temp, _ = cv2.findHomography(srcpts, dstpts, method)
                else:
                    H_temp, mask = cv2.findHomography(srcpts, dstpts, method, thresh)
                
                if H_temp is None:
                    print(f"  {method_name}: None")
                    continue
                
                # VALIDAZIONE VELOCE
                proj = cv2.perspectiveTransform(srcpts, H_temp)
                reproj_err = np.mean(np.linalg.norm(proj.squeeze() - dstpts.squeeze(), axis=1))
                
                print(f"  {method_name}: reproj_err={reproj_err:.2f}px")
                
                if reproj_err < best_reproj:
                    best_reproj = reproj_err
                    best_H = H_temp
                    
            except Exception as e:
                print(f"  {method_name}: ERROR {e}")
                continue
        
        if best_H is None:
            print("❌ ALL METHODS FAILED")
            return False
            
    except Exception as e:
        print(f"❌ DATA PREP ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== VALIDAZIONE FINALE ==========
    try:
        projpts = cv2.perspectiveTransform(normalize_calib_pts(calib_pts).reshape(-1, 1, 2), best_H)
        reproj_err = np.mean(np.linalg.norm(projpts.squeeze() - screen_pts, axis=1))
        
        print(f"\n📊 FINAL: reproj_err={reproj_err:.2f}px")
        
        if reproj_err > 3.0:
            print(f"❌ REPROJECTION TOO HIGH: {reproj_err:.2f}px")
            return False
        
        # LIVE TEST
        center_webcam = np.array([[640, 360]], dtype=np.float32).reshape(1, 1, 2)
        center_screen = cv2.perspectiveTransform(center_webcam, best_H)
        print(f"🧪 Test centro: (640,360) → ({center_screen[0][0][0]:.0f},{center_screen[0][0][1]:.0f})")
        
        homography = best_H.astype(np.float32)
        print(f"✅ ULTIMATE v9.1 SUCCESS! reproj_err={reproj_err:.2f}px")
        save_calibration()
        return True
        
    except Exception as e:
        print(f"❌ FINAL ERROR: {e}")
        return False















def project_point(raw_pt: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if homography is None or raw_pt is None:
        return None
    try:
        pt = np.array([[raw_pt]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pt, homography)[0][0]
        x = np.clip(proj[0], 30, SCREEN_WIDTH - 30)
        y = np.clip(proj[1], 30, SCREEN_HEIGHT - 30)
        return np.array([x, y], dtype=np.float32)
    except:
        return None





def smooth_trail(new_muzzle: np.ndarray, new_grip: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOOTH_TRAIL v3.0 - ULTRA-REACTIVE + ZERO JITTER
    ✅ Finestra ridotta (4-6 frame) per reattività
    ✅ Alpha base 0.85-0.95 (era 0.6-0.8)
    ✅ Velocity boost per movimenti rapidi (>15px/f)
    ✅ Prediction lookahead doppia per FPS variabili
    ✅ Dead zone ridotta (1.5px) + snap istantaneo
    """
    global gun_screen_pos, screen_grip_pos, gun_velocity, muzzle_trail, grip_trail
    
    if new_muzzle is not None:
        # APPEND SICURO
        muzzle_trail.append(new_muzzle)
        if new_grip is not None:
            grip_trail.append(new_grip)
        
        # FINESTRA ULTRA-REAKTIVA (4-6 frame max)
        window_size = min(6, max(4, len(muzzle_trail)))  # Più corta = più reattiva
        trail_array = np.array(muzzle_trail)[-window_size:]
        weighted_pos = np.mean(trail_array, axis=0)  # Weighted verso recente già intrinseco
        
        # VELOCITY AGGRESSIVA (alpha più alto)
        instant_vel = new_muzzle - gun_screen_pos
        vel_norm = np.linalg.norm(instant_vel)
        VELOCITY_ALPHA = min(0.6, 0.25 + 1.5/avg_fps)  # Più aggressivo (era 0.4 max)
        gun_velocity = (VELOCITY_ALPHA * instant_vel + 
                       (1 - VELOCITY_ALPHA) * gun_velocity)
        gun_velocity = np.clip(gun_velocity, -40, 40)  # Margine più ampio
        
        # DEAD ZONE RISTRETTA + VELOCITY BOOST
        if vel_norm > 1.5:  # Dead zone ridotta (era 2.0)
            # PREDICTION Doppia per reattività
            lookahead = 0.45 + 0.35/avg_fps  # Più lungo (era 0.3+0.2/fps)
            predicted_pos = weighted_pos + gun_velocity * lookahead
            
            # ALPHA DINAMICO: boost per velocity alta
            base_alpha = min(0.95, 0.82 + 0.4/avg_fps)  # Base più alta (era 0.8 max)
            alpha_smooth = base_alpha + min(0.12, vel_norm * 0.008)  # Boost fino +12%
            alpha_smooth = min(0.98, alpha_smooth)  # Cap per evitare overshoot
            
            # SMOOTH ULTRA-REAKTIVO
            gun_screen_pos[:] = (alpha_smooth * predicted_pos + 
                                (1 - alpha_smooth) * gun_screen_pos)
        else:
            # SNAP ISTANTANEO se stabile (zero lag)
            gun_screen_pos[:] = weighted_pos
        
        # BOUNDARY ultra-sicuri
        gun_screen_pos[0] = np.clip(gun_screen_pos[0], 30, SCREEN_WIDTH - 30)
        gun_screen_pos[1] = np.clip(gun_screen_pos[1], 30, SCREEN_HEIGHT - 30)
        
        # GRIP: finestra più corta per reattività
        if len(grip_trail) >= 2:  # Era 3, ora 2
            screen_grip_pos[:] = np.mean(np.array(grip_trail)[-2:], axis=0)
    
    return (gun_screen_pos.astype(np.int32).copy(), 
            screen_grip_pos.astype(np.int32).copy())













# ==================== WEAPON FUNCTIONS ====================

def switch_weapon(weapon_type: str):
    global current_weapon, weapon_ammo
    current_weapon = weapon_type
    weapon_ammo = WEAPONS[weapon_type].ammo

def fire_weapon(pos: np.ndarray) -> bool:
    global weapon_ammo, fire_cooldown, flash_alpha, projectiles, accuracy_shots
    if fire_cooldown > 0:
        return False
    weapon = WEAPONS[current_weapon]
    if weapon.ammo != -1:
        if weapon_ammo <= 0:
            switch_weapon('PISTOL')
            return False
        weapon_ammo -= 1
    if weapon.spread == 1:
        projectiles.append(Projectile(pos[0], pos[1], 0.0, current_weapon))
    else:
        spread_rad = math.radians(weapon.spread_angle)
        for i in range(weapon.spread):
            random_offset = random.uniform(-spread_rad, spread_rad)
            projectiles.append(Projectile(pos[0], pos[1], random_offset, current_weapon))
    fire_cooldown = weapon.fire_rate
    flash_alpha = 80 if weapon.spread > 1 else 60  # Reduced flash duration
    return True

# ==================== TARGET & POWERUP FUNCTIONS ====================

def spawn_target() -> Optional[dict]:
    global targets_spawned
    if current_level >= len(LEVELS):
        return None
    level = LEVELS[current_level]
    if 'BOSS' in level['target_types']:
        ttype = 'BOSS'
        tdata = TARGET_TYPES[ttype]
        return {
            'type': ttype, 'x': SCREEN_WIDTH // 2, 'y': 150, 'r': tdata['min_r'],
            'vy': 0, 'vx': 2.5, 'points': tdata['points'], 'hp': tdata['hp'],
            'max_hp': tdata['hp'], 'color': tdata['color'], 'glow': tdata['glow'],
            'pattern': tdata['pattern'], 'age': 0, 'rotation': 0,
            'bonus_life': tdata.get('bonus_life', False), 'penalty': tdata.get('penalty', False)
        }
    ttype = random.choice(level['target_types'])
    tdata = TARGET_TYPES[ttype]
    targets_spawned += 1
    return {
        'type': ttype, 'x': random.randint(150, SCREEN_WIDTH - 150), 'y': -100,
        'r': random.randint(tdata['min_r'], tdata['max_r']),
        'vy': random.uniform(tdata['speed_min'], tdata['speed_max']),
        'vx': random.uniform(-1.5, 1.5) if tdata['pattern'] == 'zigzag' else 0,
        'points': tdata['points'], 'hp': tdata['hp'], 'max_hp': tdata['hp'],
        'color': tdata['color'], 'glow': tdata['glow'], 'pattern': tdata['pattern'],
        'age': 0, 'rotation': 0, 'bonus_life': tdata.get('bonus_life', False),
        'penalty': tdata.get('penalty', False)
    }

def spawn_powerup(x: float, y: float):
    powerup_type = random.choice(list(POWERUP_TYPES.keys()))
    ptype = POWERUP_TYPES[powerup_type]
    powerups.append({
        'type': powerup_type,
        'x': x,
        'y': y,
        'vy': 2.0,
        'age': 0,
        'data': ptype
    })

def check_hit(target: dict, pos: np.ndarray) -> Tuple[bool, bool]:
    effective_r = target['r'] * 1.2
    dist = np.hypot(target['x'] - pos[0], target['y'] - pos[1])
    center_dist = dist / max(target['r'], 1)
    perfect = center_dist < 0.3
    return dist < effective_r, perfect

# ==================== DRAWING FUNCTIONS ====================




def draw_background(surface: pygame.Surface, level_id: int):
    if level_id < len(LEVELS):
        base_color = LEVELS[level_id]['background_color']
    else:
        base_color = (10, 20, 40)
    for i in range(SCREEN_HEIGHT):
        ratio = i / SCREEN_HEIGHT
        r = int(np.clip(base_color[0] + 20 * ratio, 0, 255))
        g = int(np.clip(base_color[1] + 30 * ratio, 0, 255))
        b = int(np.clip(base_color[2] + 50 * ratio, 0, 255))
        pygame.draw.line(surface, (r, g, b), (0, i), (SCREEN_WIDTH, i))
    for star in stars:
        star['z'] -= star['speed'] * 0.5
        if star['z'] <= 0:
            star['z'] = 100
            star['x'] = random.randint(0, SCREEN_WIDTH)
            star['y'] = random.randint(0, SCREEN_HEIGHT)
        k = 128.0 / star['z']
        px = int((star['x'] - SCREEN_WIDTH/2) * k + SCREEN_WIDTH/2)
        py = int((star['y'] - SCREEN_HEIGHT/2) * k + SCREEN_HEIGHT/2)
        if 0 <= px < SCREEN_WIDTH and 0 <= py < SCREEN_HEIGHT:
            brightness = int(np.clip(255 * (1 - star['z'] / 100), 50, 255))
            size = max(1, int(3 * k))
            color = (brightness, brightness, int(np.clip(brightness + 30, 0, 255)))
            if size == 1:
                try:
                    surface.set_at((px, py), color)
                except:
                    pass
            else:
                pygame.draw.circle(surface, color, (px, py), size)





def draw_target(surface: pygame.Surface, target: dict):
    x, y = int(target['x']), int(target['y'])
    r = int(target['r'])
    pulse = 1.0 + 0.05 * math.sin(target['age'] * 0.15)
    current_r = int(r * pulse)
    target['rotation'] = (target['rotation'] + 2) % 360
    if target['hp'] > 1:
        bar_w = current_r * 2
        bar_h = 5
        bar_x = x - bar_w // 2
        bar_y = y - current_r - 12
        pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
        hp_ratio = target['hp'] / target['max_hp']
        hp_w = int(bar_w * hp_ratio)
        hp_color = (0, 255, 0) if hp_ratio > 0.5 else (255, 200, 0) if hp_ratio > 0.25 else (255, 50, 50)
        pygame.draw.rect(surface, hp_color, (bar_x, bar_y, hp_w, bar_h))
    glow_surf = pygame.Surface((current_r*2.5, current_r*2.5), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, (*target['glow'], 50), (int(current_r*1.25), int(current_r*1.25)), int(current_r * 1.2))
    surface.blit(glow_surf, (x - int(current_r*1.25), y - int(current_r*1.25)))
    pygame.draw.circle(surface, target['color'], (x, y), current_r, 4)
    pygame.draw.circle(surface, target['glow'], (x, y), int(current_r * 0.7), 2)
    center_color = COLOR_WHITE if not target.get('penalty') else COLOR_RED
    pygame.draw.circle(surface, center_color, (x, y), int(current_r * 0.3))
    if target.get('bonus_life'):
        icon = font_small.render("♥", True, (255, 100, 100))
        surface.blit(icon, (x - icon.get_width()//2, y - 8))
    elif target.get('penalty'):
        icon = font_small.render("☠", True, (255, 50, 50))
        surface.blit(icon, (x - icon.get_width()//2, y - 8))















def draw_powerup(surface: pygame.Surface, powerup: dict):
    x, y = int(powerup['x']), int(powerup['y'])
    ptype = powerup['data']
    pulse = 1.0 + 0.15 * math.sin(powerup['age'] * 0.2)
    r = int(35 * pulse)
    
    # Glow effect (update to hexagonal)
    glow_surf = pygame.Surface((r*3, r*3), pygame.SRCALPHA)
    points = []
    for i in range(6):
        angle = math.pi * 2 * i / 6 - math.pi / 2
        px = int(r*1.5 + r * 1.3 * math.cos(angle))
        py = int(r*1.5 + r * 1.3 * math.sin(angle))
        points.append((px, py))
    pygame.draw.polygon(glow_surf, (*ptype.glow, 80), points)
    surface.blit(glow_surf, (x - int(r*1.5), y - int(r*1.5)))
    
    # Main hexagon shape
    points = []
    for i in range(6):
        angle = math.pi * 2 * i / 6 - math.pi / 2
        px = int(x + r * math.cos(angle))
        py = int(y + r * math.sin(angle))
        points.append((px, py))
    pygame.draw.polygon(surface, ptype.color, points, 4)
    
    pygame.draw.circle(surface, COLOR_WHITE, (x, y), int(r * 0.7), 2)
    icon = font.render(ptype.icon, True, COLOR_WHITE)
    surface.blit(icon, (x - icon.get_width()//2, y - icon.get_height()//2))










def draw_pistol(surface: pygame.Surface, muzzle_pos: np.ndarray, grip_pos: np.ndarray, confidence: float):
    global flash_alpha
    mx, my = int(muzzle_pos[0]), int(muzzle_pos[1])
    gx, gy = int(grip_pos[0]), int(grip_pos[1])
    if flash_alpha > 0:
        flash_alpha = max(0, flash_alpha - 20)  # Faster decay
        flash_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        flash_surf.fill((255, 255, 255, flash_alpha))
        surface.blit(flash_surf, (0, 0))
    total_len = max(20, int(np.hypot(mx - gx, my - gy)))
    dir_vec = np.array([mx - gx, my - gy], dtype=float) / total_len
    ortho = np.array([-dir_vec[1], dir_vec[0]])
    hilt_h, hilt_w = 30, 14
    hilt_center = np.array([gx, gy])
    h_p1 = hilt_center + ortho * (hilt_w/2)
    h_p2 = hilt_center - ortho * (hilt_w/2)
    h_p3 = h_p2 + dir_vec * hilt_h
    h_p4 = h_p1 + dir_vec * hilt_h
    pygame.draw.polygon(surface, (80, 80, 90),
                       [(int(h_p1[0]), int(h_p1[1])), (int(h_p2[0]), int(h_p2[1])),
                        (int(h_p3[0]), int(h_p3[1])), (int(h_p4[0]), int(h_p4[1]))])
    blade_len = total_len * 0.88
    blade_start = hilt_center + dir_vec * hilt_h * 0.8
    blade_end = blade_start + dir_vec * blade_len
    weapon = WEAPONS[current_weapon]
    laser_color = weapon.color
    for width, alpha in [(25, 25), (15, 50), (8, 90)]:
        laser_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        color = (*laser_color, alpha)
        pygame.draw.line(laser_surf, color, (int(blade_start[0]), int(blade_start[1])),
                        (int(blade_end[0]), int(blade_end[1])), width)
        surface.blit(laser_surf, (0, 0))
    pygame.draw.line(surface, laser_color, (int(blade_start[0]), int(blade_start[1])),
                    (int(blade_end[0]), int(blade_end[1])), 3)
    pygame.draw.line(surface, COLOR_WHITE, (int(blade_start[0]), int(blade_start[1])),
                    (int(blade_end[0]), int(blade_end[1])), 1)
    for rad, col in [(20, (*laser_color, 70)), (12, laser_color), (6, COLOR_WHITE)]:
        if len(col) == 4:
            s = pygame.Surface((rad*3, rad*3), pygame.SRCALPHA)
            pygame.draw.circle(s, col, (rad*1.5, rad*1.5), rad)
            surface.blit(s, (mx - rad*1.5, my - rad*1.5))
        else:
            pygame.draw.circle(surface, col, (mx, my), rad)
    if confidence > 0.5:
        cross_size = 15
        for dx, dy in [(-cross_size, 0), (cross_size, 0), (0, -cross_size), (0, cross_size)]:
            start_x = mx + (4 if dx > 0 else -4 if dx < 0 else 0)
            start_y = my + (4 if dy > 0 else -4 if dy < 0 else 0)
            pygame.draw.line(surface, laser_color, (start_x, start_y), (mx + dx, my + dy), 2)








def draw_hud(surface: pygame.Surface, score_val: int, disp_score_val: int, lives_val: int, combo_val: int, 
             level_time_val: float, level_id_val: int, accuracy_val: float, audio_lvl: int) -> int:
    if disp_score_val < score_val:
        disp_score_val = min(disp_score_val + max(1, (score_val - disp_score_val) // 8), score_val)
    
    # Barra HUD sottilissima e molto trasparente
    hud_h = 52
    hud_panel = pygame.Surface((SCREEN_WIDTH, hud_h), pygame.SRCALPHA)
    hud_panel.fill((0, 0, 0, 35))
    surface.blit(hud_panel, (0, 0))
    
    # SCORE (sinistra)
    score_text = font_small.render(str(disp_score_val), True, (255, 240, 150))
    surface.blit(score_text, (18, 10))
    
    # CUORI vite
    for i in range(lives_val):
        hx = 18 + i * 20
        hy = 30
        heart = [
            (hx+5, hy+1), (hx+3, hy-1), (hx, hy+1), (hx, hy+4),
            (hx+5, hy+8), (hx+10, hy+4), (hx+10, hy+1), (hx+7, hy-1), (hx+5, hy+1)
        ]
        pygame.draw.polygon(surface, (255, 70, 90), heart, 0)
        pygame.draw.polygon(surface, (255, 120, 140), heart, 1)
    
    # LEVEL + TIME (centro)
    if level_id_val < len(LEVELS):
        level_str = f"LV.{level_id_val+1}"
        time_left = max(0, LEVELS[level_id_val]['time_limit'] - level_time_val)
        time_str = f"{int(time_left)}s"
        
        level_text = font_small.render(level_str, True, (150, 220, 255))
        time_color = (255, 80, 80) if time_left < 10 else (220, 220, 220)
        time_text = font_small.render(time_str, True, time_color)
        
        total_w = level_text.get_width() + 12 + time_text.get_width()
        base_x = SCREEN_WIDTH // 2 - total_w // 2
        surface.blit(level_text, (base_x, 10))
        surface.blit(time_text, (base_x + level_text.get_width() + 12, 10))
    
    # ARMA (destra) con allineamento interno sicuro
    weapon = WEAPONS[current_weapon]
    name_str = weapon.name.upper()
    
    # Mantieni un margine di 20px dal bordo destro
    name_text = font_small.render(name_str, True, weapon.color)
    name_x = SCREEN_WIDTH - 20 - name_text.get_width()
    surface.blit(name_text, (name_x, 10))
    
    # AMMO sotto il nome, stesso margine
    if weapon.ammo != -1:
        ammo_str = f"{weapon_ammo}/{weapon.max_ammo}"
    else:
        ammo_str = "UNLIMITED"
    ammo_text = font_tiny.render(ammo_str, True, (210, 210, 210))
    ammo_x = SCREEN_WIDTH - 20 - ammo_text.get_width()
    surface.blit(ammo_text, (ammo_x, 30))
    
    # AUDIO bar sottile sotto ammo (allineata con ammo, altezza ridotta)
    if mic_available and mic_calibrated and state == 'play':
        meter_w = 70
        meter_h = 2
        meter_x = SCREEN_WIDTH - 20 - meter_w
        meter_y = 46
        
        pygame.draw.rect(surface, (25, 25, 30), (meter_x, meter_y, meter_w, meter_h))
        
        audio_ratio = min((audio_lvl - mic_baseline) / (mic_threshold * 1.5), 1.0)
        audio_fill = int(meter_w * max(0, audio_ratio))
        color = (255, 100, 100) if audio_lvl > mic_threshold else (100, 255, 100)
        pygame.draw.rect(surface, color, (meter_x, meter_y, audio_fill, meter_h))
    
    return disp_score_val










def create_preview(raw_frame: np.ndarray, muzzle_overlay: Optional[Tuple], 
                   grip_overlay: Optional[Tuple]) -> Optional[pygame.Surface]:
    global FRAME_SKIP_COUNTER
    FRAME_SKIP_COUNTER += 1
    if FRAME_SKIP_COUNTER % PREVIEW_SKIP != 0:
        return None
    
    bright_frame = enhance_bright(raw_frame)
    frame = bright_frame.copy()
    
    if muzzle_overlay and grip_overlay:
        # Grip overlay - sfera grande fluo verde
        cv2.circle(frame, grip_overlay, 18, (0, 255, 100), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, grip_overlay, 18, (0, 255, 200), 3, lineType=cv2.LINE_AA)
        
        # Muzzle overlay - sfera piccola fluo magenta
        cv2.circle(frame, muzzle_overlay, 11, (255, 50, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, muzzle_overlay, 11, (255, 100, 255), 2, lineType=cv2.LINE_AA)
        
        # Punta dito indice ROSA FLUO alla base grip (verso il basso)
        finger_tip = (grip_overlay[0], grip_overlay[1] + 25)  # 25px sotto grip
        cv2.circle(frame, finger_tip, 12, (255, 100, 200), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, finger_tip, 12, (255, 150, 255), 3, lineType=cv2.LINE_AA)
        
        # Linee omografia SUPER evidenziate tutte VERDI FLUO
        cv2.line(frame, grip_overlay, muzzle_overlay, (0, 255, 0), 8, lineType=cv2.LINE_AA)
        cv2.line(frame, grip_overlay, muzzle_overlay, (50, 255, 50), 6, lineType=cv2.LINE_AA)
        cv2.line(frame, grip_overlay, muzzle_overlay, (0, 255, 100), 4, lineType=cv2.LINE_AA)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_LINEAR)
    
    # Crea surface con alpha e applica trasparenza (50% opacità)
    preview_surf = pygame.surfarray.make_surface(np.rot90(rgb))
    preview_surf = preview_surf.convert_alpha()
    
    alpha_surf = pygame.Surface((PREVIEW_W, PREVIEW_H), pygame.SRCALPHA)
    alpha_surf.blit(preview_surf, (0, 0))
    alpha_surf.set_alpha(128)
    
    return alpha_surf




# ==================== MAIN LOOP ====================

running = True
cached_preview = None
target_spawn_timer = 0
space_pressed_last_frame = False

print("✅ POINT BLACK ULTIMATE v8.7 LOADED!")
print("🔧 Calibration loop fixed")
print("🎨 Visual improvements applied")
print("⚡ Performance optimizations")
print("")






try:


    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0

        if mic_available:
            audio_level = get_audio_level()
        else:
            audio_level = 0

        ret, raw_frame = cap.read()
        if not ret:
            continue

        pistol_center_raw, pistol_grip_raw, muzzle_overlay, grip_overlay = detect_realistic_pistol(raw_frame)

        proj_muzzle = None
        proj_grip = None

        if pistol_center_raw is not None and homography is not None:
            proj_muzzle = project_point(pistol_center_raw)
            proj_grip = project_point(pistol_grip_raw)
            if proj_muzzle is not None and proj_grip is not None:
                gun_screen_pos, screen_grip_pos = smooth_trail(proj_muzzle, proj_grip)

        mouse_pos = pygame.mouse.get_pos()
        keys = pygame.key.get_pressed()
        space_pressed_now = keys[pygame.K_SPACE]
        space_just_pressed = space_pressed_now and not space_pressed_last_frame
        space_pressed_last_frame = space_pressed_now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and state == 'menu':
                for btn in menu_buttons:
                    if btn.hovered and btn.enabled:
                        if btn.action == "mic_calib" and mic_available:
                            state = 'mic_calib'
                            mic_calib_state = 'silence'
                            mic_calib_start = time.time()
                            mic_silence_samples = []
                            mic_noise_samples = []
                        elif btn.action == "calib":
                            state = 'calib'
                            calib_step = 0
                            calib_pts = []
                        elif btn.action == "play" and homography is not None:
                            state = 'play'
                            current_level = 0
                            score = 0
                            displayed_score = 0
                            combo = 0
                            max_combo = 0
                            lives = 5
                            accuracy_shots = 0
                            accuracy_hits = 0
                            targets = []
                            particles = []
                            damage_numbers = []
                            powerups = []
                            projectiles = []
                            targets_spawned = 0
                            level_start_time = time.time()
                            level_complete_timer = 0.0
                            switch_weapon('PISTOL')
                        elif btn.action == "reset":
                            # RESET COMPLETO - FULL CLEANUP
                            if os.path.exists(CALIB_FILE):
                                os.remove(CALIB_FILE)
                            global calibstep, miccalibrated, pistolconfidence
                            homography = None
                            calib_pts = []
                            calibstep = 0
                            miccalibrated = False
                            pistolconfidence = 0.0  # Reset tracking
                            print("🔄 Calibration COMPLETELY reset!")
                            init_menu_buttons()
                            create_explosion(btn.rect.centerx, btn.rect.centery, (0, 200, 255), 0.7)
                            flashalpha = 120
                            break
                        create_explosion(btn.rect.centerx, btn.rect.centery, (0, 200, 255), 0.7)
                        flash_alpha = 120
                        break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if state == 'play':
                        state = 'menu'
                        init_menu_buttons()
                    elif state == 'calib' or state == 'mic_calib':
                        state = 'menu'
                        init_menu_buttons()
                    elif state == 'menu':
                        running = False
                elif event.key == pygame.K_1 and state == 'menu' and mic_available:
                    state = 'mic_calib'
                    mic_calib_state = 'silence'
                    mic_calib_start = time.time()
                    mic_silence_samples = []
                    mic_noise_samples = []
                elif event.key == pygame.K_2 and state == 'menu':
                    state = 'calib'
                    calib_step = 0
                    calib_pts = []
                elif event.key == pygame.K_3 and state == 'menu' and homography is not None:
                    state = 'play'
                    current_level = 0
                    score = 0
                    displayed_score = 0
                    combo = 0
                    max_combo = 0
                    lives = 5
                    accuracy_shots = 0
                    accuracy_hits = 0
                    targets = []
                    particles = []
                    damage_numbers = []
                    powerups = []
                    projectiles = []
                    targets_spawned = 0
                    level_start_time = time.time()
                    level_complete_timer = 0.0
                    switch_weapon('PISTOL')
                elif event.key == pygame.K_r and state == 'menu':
                    if os.path.exists(CALIB_FILE):
                        os.remove(CALIB_FILE)
                    homography = None
                    calib_pts = []
                    mic_calibrated = False
                    init_menu_buttons()
                elif event.key == pygame.K_SPACE and state == 'calib' and calib_step < 4 and pistol_center_raw is not None:
                    calib_pts.append(pistol_center_raw.tolist())
                    calib_step += 1
                    if calib_step == 4:
                        if compute_homography_pro():
                            state = 'menu'
                            init_menu_buttons()
                        else:
                            calib_step = 0
                            calib_pts = []

        if state == 'menu' and space_just_pressed:
            for btn in menu_buttons:
                if btn.hovered and btn.enabled:
                    if btn.action == "mic_calib" and mic_available:
                        state = 'mic_calib'
                        mic_calib_state = 'silence'
                        mic_calib_start = time.time()
                        mic_silence_samples = []
                        mic_noise_samples = []
                    elif btn.action == "calib":
                        state = 'calib'
                        calib_step = 0
                        calib_pts = []
                    elif btn.action == "play" and homography is not None:
                        state = 'play'
                        current_level = 0
                        score = 0
                        displayed_score = 0
                        combo = 0
                        max_combo = 0
                        lives = 5
                        accuracy_shots = 0
                        accuracy_hits = 0
                        targets = []
                        particles = []
                        damage_numbers = []
                        powerups = []
                        projectiles = []
                        targets_spawned = 0
                        level_start_time = time.time()
                        level_complete_timer = 0.0
                        switch_weapon('PISTOL')
                    elif btn.action == "reset":
                        # RESET COMPLETO - FULL CLEANUP
                        if os.path.exists(CALIB_FILE):
                            os.remove(CALIB_FILE)
                        homography = None
                        calib_pts = []
                        calibstep = 0
                        miccalibrated = False
                        pistolconfidence = 0.0  # Reset tracking
                        print("🔄 Calibration COMPLETELY reset!")
                        init_menu_buttons()
                        create_explosion(btn.rect.centerx, btn.rect.centery, (0, 200, 255), 0.7)
                        flashalpha = 120
                        break

                    create_explosion(btn.rect.centerx, btn.rect.centery, (0, 200, 255), 0.7)
                    flash_alpha = 120
                    break

        if state == 'mic_calib':
            elapsed = time.time() - mic_calib_start
            if mic_calib_state == 'silence':
                mic_silence_samples.append(audio_level)
                if elapsed >= 3.0:
                    mic_calib_state = 'noise'
                    mic_calib_start = time.time()
            elif mic_calib_state == 'noise':
                mic_noise_samples.append(audio_level)
                if elapsed >= 3.0:
                    if mic_silence_samples and mic_noise_samples:
                        avg_silence = np.mean(mic_silence_samples)
                        avg_noise = np.mean(mic_noise_samples)
                        mic_baseline = int(avg_silence * 1.2)
                        mic_threshold = int((avg_silence + avg_noise) / 2)
                        mic_threshold = max(mic_threshold, int(mic_baseline * 1.5))
                        mic_calibrated = True
                        print(f"🎤 Mic calibrated! Baseline: {mic_baseline}, Threshold: {mic_threshold}")
                    state = 'menu'
                    init_menu_buttons()

        if fire_cooldown > 0:
            fire_cooldown -= 1

        if state == 'play':
            weapon = WEAPONS[current_weapon]
            now = time.time()
            voice_triggered = False
            if mic_calibrated and mic_available and audio_level > mic_threshold:
                if now - last_voice_shot > voice_cooldown:
                    voice_triggered = True
                    last_voice_shot = now
            space_trigger = False
            if weapon.auto_fire:
                if space_pressed_now:
                    space_trigger = True
            else:
                if space_just_pressed:
                    space_trigger = True
            if voice_triggered or space_trigger:
                if fire_weapon(gun_screen_pos):
                    accuracy_shots += 1

        if state == 'play':
            if current_level >= len(LEVELS):
                state = 'menu'
                init_menu_buttons()
                continue
            level = LEVELS[current_level]
            level_time = time.time() - level_start_time
            if level_time > level['time_limit']:
                lives -= 1
                if lives <= 0:
                    state = 'menu'
                    init_menu_buttons()
                else:
                    current_level += 1
                    if current_level < len(LEVELS):
                        targets = []
                        particles = []
                        damage_numbers = []
                        powerups = []
                        projectiles = []
                        targets_spawned = 0
                        level_start_time = time.time()
                        level_complete_timer = 0.0
                    else:
                        state = 'menu'
                        init_menu_buttons()
                continue
            target_spawn_timer += 1
            if (target_spawn_timer > level['spawn_rate'] and 
                len(targets) < level['max_targets'] and 
                targets_spawned < level['targets_to_spawn']):
                new_target = spawn_target()
                if new_target:
                    targets.append(new_target)
                target_spawn_timer = 0
            for t in targets[:]:
                if t['pattern'] == 'zigzag':
                    t['x'] += t['vx']
                    if t['x'] < 100 or t['x'] > SCREEN_WIDTH - 100:
                        t['vx'] *= -1
                elif t['pattern'] == 'boss':
                    t['x'] += t['vx']
                    if t['x'] < 200 or t['x'] > SCREEN_WIDTH - 200:
                        t['vx'] *= -1
                t['y'] += t['vy']
                t['age'] += 1
                if t['y'] > SCREEN_HEIGHT + 200:
                    targets.remove(t)
                    if not t.get('penalty') and t['type'] != 'BOSS':
                        lives -= 1
                        combo = 0
            for proj in projectiles:
                if not proj.update():
                    projectiles.remove(proj)
                    continue
                for t in targets[:]:
                    hit, perfect = check_hit(t, np.array([proj.x, proj.y]))
                    if hit:
                        t['hp'] -= proj.weapon_stats.damage
                        proj.hit = True
                        if t['hp'] <= 0:
                            if t.get('bonus_life'):
                                lives = min(lives + 1, 5)
                                damage_numbers.append(DamageNumber(t['x'], t['y'], 1, False))
                            if not t.get('penalty'):
                                combo += 1
                                max_combo = max(max_combo, combo)
                                bonus = 25 if perfect else 0
                                total_points = t['points'] + bonus + (combo // 3) * 10
                                score += total_points
                                accuracy_hits += 1
                                damage_numbers.append(DamageNumber(t['x'], t['y'], total_points, perfect))
                                if random.random() < level['powerup_chance']:
                                    spawn_powerup(t['x'], t['y'])
                            else:
                                score = max(0, score + t['points'])
                                combo = 0
                                lives = max(0, lives - 1)
                            create_explosion(t['x'], t['y'], t['color'], 1.0)
                            targets.remove(t)
                        else:
                            create_explosion(t['x'], t['y'], t['color'], 0.4)
                        break
            for p in powerups[:]:
                p['y'] += p['vy']
                p['age'] += 1
                dist = np.hypot(p['x'] - gun_screen_pos[0], p['y'] - gun_screen_pos[1])
                if dist < 50:
                    ptype = p['data']
                    if ptype.effect_type == 'health':
                        old_lives = lives
                        lives = min(lives + ptype.value, 5)
                        if lives > old_lives:
                            damage_numbers.append(DamageNumber(p['x'], p['y'], 1, False))
                    elif ptype.effect_type == 'weapon' and ptype.weapon_type:
                        switch_weapon(ptype.weapon_type)
                        weapon_ammo = WEAPONS[ptype.weapon_type].max_ammo
                    create_explosion(p['x'], p['y'], ptype.color, 0.6)
                    powerups.remove(p)
                elif p['y'] > SCREEN_HEIGHT + 100:
                    powerups.remove(p)



            particles[:] = [p for p in particles if p.update()]  # O usa iterator
            # Meglio:
            i = 0
            while i < len(particles):
                if not particles[i].update():
                    del particles[i]
                else:
                    i += 1




            damage_numbers = [d for d in damage_numbers if d.update()]



            if targets_spawned >= level['targets_to_spawn'] and len(targets) == 0:
                if level_complete_timer == 0.0:
                    level_complete_timer = time.time()
                elif time.time() - level_complete_timer > level_complete_delay:
                    current_level += 1
                    if current_level < len(LEVELS):
                        targets = []
                        particles = []
                        damage_numbers = []
                        powerups = []
                        projectiles = []
                        targets_spawned = 0
                        level_start_time = time.time()
                        level_complete_timer = 0.0
                    else:
                        state = 'menu'
                        init_menu_buttons()
            if lives <= 0:
                state = 'menu'
                init_menu_buttons()
        elif state == 'menu':
            cursor_pos = mouse_pos if pygame.mouse.get_focused() else (int(gun_screen_pos[0]), int(gun_screen_pos[1]))
            for btn in menu_buttons:
                btn.check_hover(cursor_pos)
            menu_buttons[0].enabled = mic_available
            menu_buttons[2].enabled = (homography is not None)

        preview_surf = create_preview(raw_frame, muzzle_overlay, grip_overlay)
        if preview_surf:
            cached_preview = preview_surf

        current_fps = clock.get_fps()
        FPS_HISTORY.append(max(current_fps, 30))
        avg_fps = np.mean(list(FPS_HISTORY))

        screen.fill(COLOR_BLACK)
        shake_offset = (0, 0)
        if screen_shake > 0:
            shake_offset = (random.randint(-screen_shake, screen_shake), 
                        random.randint(-screen_shake, screen_shake))
            screen_shake -= 1
        temp_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        if state == 'menu':
            draw_background(temp_surface, 0)

            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            temp_surface.blit(overlay, (0, 0))

            center_x = SCREEN_WIDTH // 2
            y = 60  # cursore verticale

            def blit_centered(surface, text_surf, y):
                surface.blit(text_surf, (center_x - text_surf.get_width() // 2, y))
                return y + text_surf.get_height()

            # === TITLE ===
            title = font_title.render("POINT BLACK", True, (0, 220, 255))
            y = blit_centered(temp_surface, title, y) + 10

            subtitle = font_small.render(
                "Develop by Indecenti", True, (100, 200, 255)
            )
            y = blit_centered(temp_surface, subtitle, y) + 25

            # === MIC STATUS ===
            if mic_calibrated:
                mic_text = f"Mic Calibrated (Threshold: {mic_threshold})"
                mic_color = (100, 255, 100)
            elif mic_available:
                mic_text = "Mic not calibrated – Click MIC CALIBRATION"
                mic_color = (255, 200, 100)
            else:
                mic_text = "Microphone not available"
                mic_color = (255, 100, 100)

            mic_status = font_tiny.render(mic_text, True, mic_color)
            y = blit_centered(temp_surface, mic_status, y) + 30

            # === STATS (solo se presenti) ===
            if score > 0:
                stats = [
                    f"FINAL SCORE: {score}",
                    f"MAX COMBO: x{max_combo}",
                    f"ACCURACY: {(accuracy_hits / max(accuracy_shots, 1) * 100):.1f}%"
                ]
                for stat in stats:
                    stat_text = font_small.render(stat, True, COLOR_YELLOW)
                    y = blit_centered(temp_surface, stat_text, y) + 8
                y += 25

            # === MENU BUTTONS ===
            button_start_y = y
            for btn in menu_buttons:
                btn.rect.centerx = center_x
                btn.rect.y = y
                btn.draw(temp_surface)
                y += btn.rect.height + 14

            y += 20

            # === INSTRUCTIONS ===
            inst_text = font_tiny.render(
                "MOUSE/SPACE Select | 1 MicCal | 2 HandCal | 3 Play | R Reset | ESC Exit",
                True,
                (150, 200, 255)
            )
            blit_centered(temp_surface, inst_text, min(y, SCREEN_HEIGHT - 40))

            # === CURSOR + GUN ===
            pygame.draw.circle(temp_surface, (255, 255, 100), mouse_pos, 8, 2)

            if pistol_confidence > MIN_PISTOL_CONFIDENCE:
                draw_pistol(
                    temp_surface,
                    gun_screen_pos,
                    screen_grip_pos,
                    pistol_confidence
                )

        
        
        
        
        elif state == 'mic_calib':
            draw_background(temp_surface, 0)
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            temp_surface.blit(overlay, (0, 0))
            title = font_subtitle.render("MICROPHONE CALIBRATION", True, (255, 200, 0))
            temp_surface.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 100))
            if mic_calib_state == 'silence':
                elapsed = time.time() - mic_calib_start
                remaining = 3.0 - elapsed
                inst = font.render("BE SILENT!", True, (100, 255, 100))
                temp_surface.blit(inst, (SCREEN_WIDTH // 2 - inst.get_width() // 2, 200))
                timer_text = font_title.render(f"{int(remaining)+1}", True, COLOR_WHITE)
                temp_surface.blit(timer_text, (SCREEN_WIDTH // 2 - timer_text.get_width() // 2, 300))
                bar_w = 600
                bar_h = 40
                bar_x = SCREEN_WIDTH // 2 - bar_w // 2
                bar_y = 450
                pygame.draw.rect(temp_surface, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
                fill = min(int(bar_w * audio_level / 3000), bar_w)
                pygame.draw.rect(temp_surface, (100, 255, 100), (bar_x, bar_y, fill, bar_h))
                level_text = font_small.render(f"Audio Level: {audio_level}", True, COLOR_WHITE)
                temp_surface.blit(level_text, (SCREEN_WIDTH // 2 - level_text.get_width() // 2, 510))
            elif mic_calib_state == 'noise':
                elapsed = time.time() - mic_calib_start
                remaining = 3.0 - elapsed
                inst = font.render("MAKE NOISE! SHOUT!", True, (255, 100, 100))
                temp_surface.blit(inst, (SCREEN_WIDTH // 2 - inst.get_width() // 2, 200))
                timer_text = font_title.render(f"{int(remaining)+1}", True, COLOR_WHITE)
                temp_surface.blit(timer_text, (SCREEN_WIDTH // 2 - timer_text.get_width() // 2, 300))
                bar_w = 600
                bar_h = 40
                bar_x = SCREEN_WIDTH // 2 - bar_w // 2
                bar_y = 450
                pygame.draw.rect(temp_surface, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h))
                fill = min(int(bar_w * audio_level / 3000), bar_w)
                pygame.draw.rect(temp_surface, (255, 100, 100), (bar_x, bar_y, fill, bar_h))
                level_text = font_small.render(f"Audio Level: {audio_level}", True, COLOR_WHITE)
                temp_surface.blit(level_text, (SCREEN_WIDTH // 2 - level_text.get_width() // 2, 510))
            esc_text = font_tiny.render("Press ESC to cancel", True, (200, 200, 200))
            temp_surface.blit(esc_text, (SCREEN_WIDTH // 2 - esc_text.get_width() // 2, 600))
        elif state == 'calib':
            draw_background(temp_surface, 0)
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            temp_surface.blit(overlay, (0, 0))
            title = font_subtitle.render(f"HAND CALIBRATION {calib_step + 1}/4", True, (255, 200, 0))
            temp_surface.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 50))
            inst = font.render(f"Point at: {calib_labels[calib_step]}", True, COLOR_WHITE)
            temp_surface.blit(inst, (SCREEN_WIDTH // 2 - inst.get_width() // 2, 120))
            inst2 = font_small.render("Press SPACE when ready | ESC to cancel", True, (200, 200, 200))
            temp_surface.blit(inst2, (SCREEN_WIDTH // 2 - inst2.get_width() // 2, 170))
            for i, (label, pt) in enumerate(zip(calib_labels, screen_pts)):
                if i < calib_step:
                    color = COLOR_NEON_GREEN
                    pulse_r = 40
                elif i == calib_step:
                    pulse_r = 40 + 8 * math.sin(time.time() * 5)
                    color = COLOR_NEON_PINK
                else:
                    color = (100, 100, 100)
                    pulse_r = 35
                for r in [int(pulse_r), int(pulse_r * 0.7), int(pulse_r * 0.4)]:
                    pygame.draw.circle(temp_surface, color, (int(pt[0]), int(pt[1])), r, 4)
                pygame.draw.circle(temp_surface, COLOR_WHITE, (int(pt[0]), int(pt[1])), 10)
                if i < calib_step:
                    check_text = font.render("✓", True, COLOR_NEON_GREEN)
                    temp_surface.blit(check_text, (int(pt[0]) - check_text.get_width()//2, int(pt[1]) - check_text.get_height()//2))
        elif state == 'play':
            draw_background(temp_surface, current_level)
            for t in targets[:]:
                draw_target(temp_surface, t)
            for p in powerups[:]:
                draw_powerup(temp_surface, p)
            for proj in projectiles:
                proj.draw(temp_surface)
            for p in particles:
                p.draw(temp_surface)
            for d in damage_numbers:
                d.draw(temp_surface)
            accuracy = (accuracy_hits / max(accuracy_shots, 1)) * 100
            displayed_score = draw_hud(temp_surface, score, displayed_score, lives, combo, 
                                    level_time, current_level, accuracy, audio_level)
            if pistol_confidence > MIN_PISTOL_CONFIDENCE:
                draw_pistol(temp_surface, gun_screen_pos, screen_grip_pos, pistol_confidence)

        screen.blit(temp_surface, shake_offset)

        if cached_preview:
            preview_alpha = pygame.Surface((PREVIEW_W + 6, PREVIEW_H + 6), pygame.SRCALPHA)
            preview_alpha.fill((0, 0, 0, 100))
            screen.blit(preview_alpha, (preview_rect.x - 3, preview_rect.y - 3))
            border_rect = pygame.Rect(preview_rect.x - 2, preview_rect.y - 2, preview_rect.w + 4, preview_rect.h + 4)
            pygame.draw.rect(screen, (0, 180, 255), border_rect, 2, border_radius=4)
            screen.blit(cached_preview, preview_rect.topleft)

        pygame.display.flip()

except KeyboardInterrupt:
    print("Uscita forzata")
finally:  # QUESTO È CRITICO - SI ESEGUE SEMPRE
    print("Cleanup risorse...")
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if 'stream' in locals() and stream:
        stream.stop_stream()
        stream.close()
    if 'audio' in locals():
        audio.terminate()
    pygame.quit()
    print("Risorse liberate!")














if stream:
    stream.stop_stream()
    stream.close()
if audio:
    audio.terminate()
cap.release()
pygame.quit()

print("\n" + "=" * 80)
print(f"  GAME OVER - Final Score: {score} | Max Combo: x{max_combo}")
if accuracy_shots > 0:
    print(f"  Accuracy: {(accuracy_hits/accuracy_shots*100):.1f}% ({accuracy_hits}/{accuracy_shots})")
print("=" * 80)
print("  Thanks for playing! Develop by Indecenti!")
print("=" * 80)
