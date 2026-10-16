"""
Synthetic Data Generator for AthleteIQ.
Generates:
1. Three peer-reviewed Sports Science research PDFs using ReportLab.
2. 30-day realistic GPS Telemetry CSV for cricket bowlers.
3. Physiotherapist medical clinical report for Player 9.
"""

import os
import random
import pandas as pd
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def create_research_pdf(filename, title, authors, journal, abstract_text, sections):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=21,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=10
    )
    author_style = ParagraphStyle(
        'PaperAuthor',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=12
    )
    journal_style = ParagraphStyle(
        'JournalRef',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0284C7'),
        spaceAfter=14
    )
    abstract_heading = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=8,
        spaceAfter=4
    )
    abstract_body = ParagraphStyle(
        'AbstractBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=14
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=8
    )

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(authors, author_style))
    story.append(Paragraph(journal, journal_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284C7'), spaceAfter=12))
    
    story.append(Paragraph("ABSTRACT", abstract_heading))
    story.append(Paragraph(abstract_text, abstract_body))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=12))

    for heading, text_blocks in sections:
        story.append(Paragraph(heading, section_heading))
        for block in text_blocks:
            story.append(Paragraph(block, body_style))
        story.append(Spacer(1, 6))

    doc.build(story)
    print(f"Generated PDF: {filename}")


def generate_all_papers():
    base_dir = os.path.join("data", "public_papers")
    
    # Paper 1: Fast Bowling Lumbar Stress Fractures
    paper1_path = os.path.join(base_dir, "fast_bowling_lumbar_stress.pdf")
    p1_title = "Lumbar Bone Stress Injuries and Spondylolysis in Cricket Fast Bowlers: Biomechanical Etiology and Clinical Management"
    p1_authors = "Dr. Marcus Vance, FACSEP; Dr. Sophie Thornton, PhD; Prof. Alistair Finch, MD"
    p1_journal = "Journal of Cricket Sports Medicine & Science, 2024; Vol 38(4): 412-428 | DOI: 10.1016/j.jcsm.2024.03.011"
    p1_abstract = (
        "Lumbar bone stress injuries (BSI), notably spondylolysis of the pars interarticularis (frequently at L4 and L5), "
        "represent the single most debilitating career-threatening pathology in elite fast bowlers. "
        "This prospective study evaluates 84 professional fast bowlers across 3 domestic seasons. "
        "Bowlers utilizing a mixed delivery action exhibited a 3.4-fold elevated incidence of contralateral pars stress edema (p < 0.001) "
        "compared to pure side-on or front-on bowlers. Peak ground reaction forces during front foot contact (FFC) ranged from 5.2 to 8.9 times "
        "body weight. When repetitive front-foot impact coincides with excessive trunk lateral flexion (>38 degrees) and lumbar hyperextension, "
        "the tensile and shear forces placed across the neural arch exceed bone fatigue remodeling capacity. "
        "Early 3T MRI detection of fluid-sensitive T2/STIR hyperintensity allows conservative recovery within 8-12 weeks, "
        "whereas progress to cortical fracture requires 6-9 months of complete bowling cessation."
    )
    p1_sections = [
        ("1. Anatomical Vulnerability of the Pars Interarticularis", [
            "The pars interarticularis is the bony bridge between the superior and inferior articular processes of the vertebra. "
            "In cricket fast bowling, the delivery stride demands rapid deceleration, spinal rotation, hyperextension, and contralateral lateral flexion.",
            "During right-arm delivery, the bowler's non-bowling (left) foot strikes the crease while the lumbar spine violently rotates and laterally flexes "
            "to the left. This kinematics subjects the contralateral (right-sided) L4 and L5 pars to intense compressive and shear loads. "
            "Repeated microtrauma leads to microcrack accumulation, cellular uncoupling of osteoblastic repair, and resultant bone marrow edema."
        ]),
        ("2. Front Foot Contact Kinematics and Ground Reaction Forces", [
            "During the delivery stride, front foot contact (FFC) generates vertical ground reaction forces (VGRF) reaching 6.0 to 9.0 body weight (BW) "
            "and braking forces up to 3.5 BW within 25-40 milliseconds.",
            "Fast bowlers who maintain an extended or braced front knee transmit higher ground reaction impulse directly through the kinetic chain to the pelvic "
            "ring and lumbar spine. While a rigid front knee enhances ball release velocity through energy transfer, without adequate trunk core musculature "
            "it causes severe micro-instability and stress concentration at the L5-S1 junction."
        ]),
        ("3. Clinical Presentation, Stork Test, and Imaging Diagnostic Pathways", [
            "Clinical presentation typically features insidious, unilateral lower back ache exacerbated by lumbar extension and rotational maneuvers. "
            "The single-leg hyperextension test (Stork Test) on the ipsilateral standing leg reproduces focal pain in over 85% of acute pars stress lesions.",
            "Conventional radiography (X-rays) lacks sensitivity in early-stage bone stress reactions (sensitivity < 32%). "
            "Magnetic Resonance Imaging (3T MRI) with T2 fat-suppressed (STIR) sequences is the gold standard, identifying bone marrow edema (Grade 1-2) "
            "well before fracture line cortical defect visualization on thin-slice CT (Grade 3-4)."
        ]),
        ("4. Return-to-Play and Workload Progression Protocols", [
            "Conservative rehabilitation requires immediate unloading from bowling activities while sustaining cardiovascular fitness via off-feet conditioning "
            "(wattbike, swimming). Return-to-bowling must follow a graded 6-phase criteria-based pathway:",
            "Phase 1: Pain-free daily activities and neutral spine motor control (Weeks 1-4). "
            "Phase 2: Lumbopelvic dissociation and posterior chain isometric strengthening (Weeks 4-6). "
            "Phase 3: Standing walk-through drills and bowling mechanics modification (Weeks 6-8). "
            "Phase 4: Run-up progressions at 50-75% velocity with bowling volume capped at 12-18 balls per session (Weeks 8-10). "
            "Phase 5: Match-intensity spells (100% velocity) bowling maximum 4-5 overs per spell with 48 hours recovery between sessions (Weeks 10-12). "
            "Phase 6: Full competitive clearance with strict Acute:Chronic Workload Ratio maintenance between 0.8 and 1.25."
        ])
    ]
    create_research_pdf(paper1_path, p1_title, p1_authors, p1_journal, p1_abstract, p1_sections)

    # Paper 2: Acute Chronic Workload Ratio
    paper2_path = os.path.join(base_dir, "acute_chronic_workload_cricket.pdf")
    p2_title = "The Acute:Chronic Workload Ratio (ACWR) and Bowling Injury Prevention in Elite Cricket: Evidence from Multi-Day and T20 Formats"
    p2_authors = "Dr. Timothy J. Gabbett, PhD; Dr. Karen H. Hulin, MSc; Warren Blanch, BPhysio"
    p2_journal = "British Journal of Sports Medicine & Cricket Performance Review, 2023; Vol 57(12): 789-801 | DOI: 10.1136/bjsports-2023-104922"
    p2_abstract = (
        "Monitoring training load is paramount in mitigating soft tissue strains and bone stress injuries in elite cricket bowlers. "
        "The Acute:Chronic Workload Ratio (ACWR) models the fatigue induced by recent training (acute load: rolling 7-day total) "
        "against the fitness built over time (chronic load: rolling 28-day average). In this longitudinal analysis of 120 international "
        "and first-class pace bowlers over 4,200 player-weeks, an ACWR between 0.80 and 1.30 was defined as the 'Sweet Spot', "
        "where injury risk remained lowest (<4.2% per 1,000 bowling hours). Conversely, when ACWR exceeded 1.50 (the 'Danger Zone'), "
        "the relative risk of bowling injury increased by 2.8 to 4.1 times in the subsequent 7 to 21 days. "
        "Spikes in bowling delivery volume without adequate chronic conditioning foundation represent the primary modifiable trigger for lumbar stress."
    )
    p2_sections = [
        ("1. Mathematical Formulation and Decoupling Bowling Loads", [
            "Acute workload is calculated as the sum of bowling deliveries (and GPS PlayerLoad units) performed over the preceding 7 days (Days t-6 to t). "
            "Chronic workload is calculated as the 28-day rolling average: (Sum of 28 days of load) / 4. "
            "ACWR is the quotient: ACWR = Acute Workload / Chronic Workload.",
            "In modern sports science, Exponentially Weighted Moving Averages (EWMA) are also utilized to assign greater decay weight to recent days. "
            "Regardless of rolling average or EWMA methodology, sudden weekly bowling volume increases (>15-20% week-on-week) reliably destabilize the ratio."
        ]),
        ("2. The Sweet Spot vs Danger Zone Thresholds", [
            "Statistical modeling defines three distinct physiological risk zones for cricket pace bowlers:",
            "1. Under-prepared / Sub-optimal Zone (ACWR < 0.80): Fitness decay, loss of tissue resilience, elevated risk upon sudden match re-introduction.",
            "2. High Performance Sweet Spot (ACWR 0.80 - 1.30): Optimal balance of acute stimulation and chronic physical conditioning; lowest injury hazard.",
            "3. Elevated Risk Zone (ACWR 1.30 - 1.49): Early warning indicator requiring monitoring of sleep, subjective wellness, and RPE scores.",
            "4. Critical Danger Zone (ACWR >= 1.50): Workload spike threshold. In our cohort, 78% of lumbar stress reactions occurred within 14 days of an ACWR spike >= 1.55."
        ]),
        ("3. Match Format Interactions: T20 vs Multi-Day Demands", [
            "The compressed modern schedule forces rapid transitions between multi-day cricket (where a bowler may bowl 30-45 overs across 4 days) "
            "and franchise T20 tournaments (bowling only 4 overs per match).",
            "When bowlers transition from T20 leagues directly into red-ball test or shield cricket without a 4-week progressive chronic volume ramp, "
            "their chronic workload base is severely depressed (often only 12-16 overs/week). Bowling 25+ overs in a single innings causes an instantaneous "
            "ACWR spike to 2.1+, precipitating hamstring strains and spinal stress reactions."
        ]),
        ("4. Practical Guidelines for Pace Bowler Load Management", [
            "To safeguard bowlers, high-performance coaching staff must implement weekly bowling thresholds:",
            "- Do not exceed a 10% to 15% increase in total balls bowled from one 7-day microcycle to the next.",
            "- Ensure a minimum of 24-48 hours rest between maximal bowling workloads (>36 deliveries at >90% effort).",
            "- If ACWR exceeds 1.50 due to unavoidable match demands, mandate an immediate deload microcycle: restrict net bowling to walk-throughs, "
            "prioritize manual physiotherapy soft-tissue therapy, and track daily lumbar palpation scores."
        ])
    ]
    create_research_pdf(paper2_path, p2_title, p2_authors, p2_journal, p2_abstract, p2_sections)

    # Paper 3: Shoulder Counter-Rotation Biomechanics
    paper3_path = os.path.join(base_dir, "shoulder_counter_rotation_biomechanics.pdf")
    p3_title = "Shoulder Counter-Rotation and Spinal Torsional Torque in Cricket Fast Bowling Delivery Actions"
    p3_authors = "Prof. Bruce Elliott, PhD; Dr. Jacqueline Alderson, PhD; Dr. René Ferdinands, PhD"
    p3_journal = "Sports Biomechanics & Orthopaedic Engineering, 2023; Vol 22(8): 504-522 | DOI: 10.1080/14763141.2023.2189033"
    p3_abstract = (
        "Fast bowling delivery actions are traditionally categorized based on pelvic and shoulder alignment at back foot contact (BFC) "
        "and front foot contact (FFC) into: Side-on, Front-on, Semi-open, and Mixed. A 'mixed action' occurs when a bowler adopts a front-on "
        "pelvic alignment at BFC while counter-rotating the shoulder axis into a more side-on position prior to FFC, or vice versa. "
        "Three-dimensional motion analysis (250 Hz optoelectronic cameras and instrumented force plates) of 62 first-class pace bowlers demonstrated "
        "that shoulder counter-rotation exceeding 30 degrees generates extreme twisting torsion and asymmetric facet joint impingement at L4-S1. "
        "Bowlers displaying counter-rotation >35 degrees demonstrated a 5.2-fold higher prevalence of symptomatic spondylolysis than those maintaining "
        "counter-rotation under 20 degrees. Biomechanical technique retraining aimed at aligning pelvic and thoracic rotation vectors is vital."
    )
    p3_sections = [
        ("1. Classification of Fast Bowling Delivery Techniques", [
            "Biomechanists define bowling technique through shoulder and pelvic alignment angles relative to the bowling crease:",
            "1. Pure Side-On: Shoulder angle < 110 degrees, pelvic angle < 110 degrees at BFC; hips and shoulders remain parallel throughout delivery.",
            "2. Pure Front-On: Shoulder angle > 130 degrees, pelvic angle > 130 degrees at BFC; minimal trunk twisting, lower lumbar shear.",
            "3. Semi-Open: Pelvic and shoulder angles between 110 and 130 degrees with aligned vectors.",
            "4. Mixed Action: Characterized by severe discordance between hips and shoulders. Specifically, hips open towards the batsman (>130 degrees) "
            "while the thoracic shoulder line actively twists backwards into side-on (<100 degrees)."
        ]),
        ("2. Definition and Calculation of Counter-Rotation", [
            "Counter-rotation is mathematically defined as the maximum shoulder realignment angle occurring between back foot contact and front foot contact:",
            "Counter-Rotation (deg) = Shoulder Angle at Back Foot Contact (deg) - Minimum Shoulder Angle Achieved Before Delivery (deg).",
            "Values under 20 degrees are considered biomechanically benign and safe. "
            "Values between 20 and 30 degrees indicate moderate spinal torque requiring close physiological monitoring. "
            "Values exceeding 30 degrees, and particularly >35-40 degrees, are classified as pathological mixed actions with extreme lumbar shear forces."
        ]),
        ("3. Kinetic Mechanics of Lumbar Spine Shear Stress", [
            "When the lumbar spine is simultaneously subjected to high torsional rotation, lateral hyperextension, and vertical deceleration forces at FFC, "
            "the inferior articular processes of the upper vertebrae impact violently against the lamina and pars interarticularis of the lower vertebrae.",
            "This creates asymmetric mechanical loading. For a right-arm fast bowler who counter-rotates his shoulders back to the right before snapping forwards, "
            "the contralateral right pars interarticularis of L4 and L5 suffers maximal compressive crushing forces. Fatigue failure ensues rapidly under high match overs."
        ]),
        ("4. Biomechanical Modification and Coaching Interventions", [
            "Correcting a mixed bowling action requires structured kinematic recalibration:",
            "- Align run-up trajectory: Avoid crossing over the center line of the pitch during delivery stride.",
            "- Stabilize back foot placement: Ensure the back foot lands parallel to the crease (if side-on) or pointing down the pitch (if front-on).",
            "- Non-bowling arm pathway: Cue the lead arm to pull down in the sagittal plane rather than swinging wide across the torso, which drags the thoracic spine into excessive rotation.",
            "- Video-based biofeedback: High-speed 240 fps cameras positioned directly behind and side-on allow coaches and sports scientists to measure shoulder angle discrepancies weekly."
        ])
    ]
    create_research_pdf(paper3_path, p3_title, p3_authors, p3_journal, p3_abstract, p3_sections)


def generate_player_telemetry():
    raw_csv_path = os.path.join("data", "private_telemetry", "raw_csvs", "player_daily_metrics.csv")
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
    
    players = [
        {"id": "Player 9", "name": "Arjun Patel", "role": "Right-arm Fast Bowler (Express)", "base_acute": 18, "base_chronic": 16},
        {"id": "Player 4", "name": "Liam Smith", "role": "Right-arm Fast-Medium Bowler", "base_acute": 14, "base_chronic": 15},
        {"id": "Player 12", "name": "Rashid Khan", "role": "Right-arm Leg Spinner", "base_acute": 22, "base_chronic": 24},
        {"id": "Player 7", "name": "Mitchell Stark", "role": "Left-arm Fast Bowler", "base_acute": 16, "base_chronic": 18},
    ]
    
    start_date = datetime(2026, 9, 1)
    rows = []
    
    random.seed(42)
    
    for day in range(30):
        current_date = start_date + timedelta(days=day)
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Day 21 to Day 25: Match fixture where Player 9 is heavily overbowled!
        is_heavy_match_window = (20 <= day <= 24)
        
        for p in players:
            p_id = p["id"]
            p_name = p["name"]
            p_role = p["role"]
            
            if p_id == "Player 9":
                if day < 15:
                    # Normal conditioning phase
                    session = random.choice(["Net Session", "Strength & Conditioning", "Rest Day", "Net Session"])
                    overs = random.choice([0, 4, 6, 0, 5]) if session != "Rest Day" else 0
                    rpe = random.randint(4, 7) if overs > 0 else 2
                    lumbar_soreness = random.randint(1, 3)
                    sleep = round(random.uniform(7.5, 8.5), 1)
                    acute_workload = round(14.0 + random.uniform(-2, 3), 1)
                    chronic_workload = round(16.5 + random.uniform(-1, 1), 1)
                elif 15 <= day < 20:
                    # Light buildup
                    session = "Net Session" if day % 2 == 0 else "Strength & Conditioning"
                    overs = 6 if session == "Net Session" else 0
                    rpe = 6 if overs > 0 else 4
                    lumbar_soreness = random.randint(2, 4)
                    sleep = round(random.uniform(7.0, 8.0), 1)
                    acute_workload = round(18.0 + random.uniform(-1, 2), 1)
                    chronic_workload = 15.0
                elif is_heavy_match_window:
                    # Match spike! Overbowled
                    session = "First-Class 4-Day Match"
                    overs = random.choice([9, 10, 8, 9, 0])
                    rpe = random.randint(8, 10)
                    lumbar_soreness = 5 if day == 20 else (6 if day == 21 else (8 if day == 22 else 9))
                    sleep = round(random.uniform(5.5, 6.8), 1)
                    acute_workload = round(28.0 + (day - 20) * 2.5, 1) # Acute spikes up to 36+ overs!
                    chronic_workload = 16.2
                else:
                    # Post injury onset: medical deload
                    session = "Medical Rehab & Physio"
                    overs = 0
                    rpe = 3
                    lumbar_soreness = max(4, 9 - (day - 24))
                    sleep = round(random.uniform(7.8, 8.8), 1)
                    acute_workload = round(max(8.0, 32.0 - (day - 24) * 4.5), 1)
                    chronic_workload = 17.0

                acwr = round(acute_workload / max(1.0, chronic_workload), 2)
                run_up_vel = round(random.uniform(26.5, 29.8), 1) if overs > 0 else 0.0
                dist_km = round(random.uniform(8.5, 13.8), 2) if overs > 0 else round(random.uniform(2.0, 4.5), 2)
                hsr_m = round(dist_km * random.uniform(140, 210), 0) if overs > 0 else round(random.uniform(50, 150), 0)
                sprint_m = round(hsr_m * 0.35, 0) if overs > 0 else 0
                load = round(dist_km * 75 + overs * 22, 1)
            else:
                # Other normal players
                session = random.choice(["Net Session", "Strength & Conditioning", "Rest Day", "Match"])
                overs = random.randint(3, 7) if session in ["Net Session", "Match"] else 0
                rpe = random.randint(4, 8) if overs > 0 else random.randint(2, 4)
                lumbar_soreness = random.randint(1, 3)
                sleep = round(random.uniform(7.0, 8.5), 1)
                acute_workload = round(p["base_acute"] + random.uniform(-3, 4), 1)
                chronic_workload = round(p["base_chronic"] + random.uniform(-1, 2), 1)
                acwr = round(acute_workload / chronic_workload, 2)
                run_up_vel = round(random.uniform(24.0, 28.5), 1) if (overs > 0 and "Fast" in p_role) else (round(random.uniform(12.0, 16.0), 1) if overs > 0 else 0.0)
                dist_km = round(random.uniform(6.0, 11.0), 2) if overs > 0 else round(random.uniform(2.0, 4.0), 2)
                hsr_m = round(dist_km * 120, 0) if overs > 0 else 50
                sprint_m = round(hsr_m * 0.25, 0) if overs > 0 else 0
                load = round(dist_km * 70 + overs * 18, 1)

            rows.append({
                "date": date_str,
                "player_id": p_id,
                "player_name": p_name,
                "role": p_role,
                "session_type": session,
                "overs_bowled": overs,
                "balls_bowled": overs * 6,
                "total_distance_km": dist_km,
                "high_speed_running_m": int(hsr_m),
                "sprint_distance_m": int(sprint_m),
                "player_load": load,
                "run_up_velocity_kmh": run_up_vel,
                "acute_workload": acute_workload,
                "chronic_workload": chronic_workload,
                "acwr": acwr,
                "rpe": rpe,
                "sleep_hours": sleep,
                "lumbar_soreness": lumbar_soreness,
                "hamstring_soreness": random.randint(1, 3)
            })

    df = pd.DataFrame(rows)
    df.to_csv(raw_csv_path, index=False)
    print(f"Generated Telemetry CSV: {raw_csv_path} ({len(df)} rows)")


def generate_physio_notes():
    med_dir = os.path.join("data", "private_telemetry", "medical_reports")
    os.makedirs(med_dir, exist_ok=True)
    report_path = os.path.join(med_dir, "player_9_physio_notes.txt")
    
    content = """CONFIDENTIAL ATHLETE MEDICAL & PHYSIOTHERAPY REPORT
ATHLETE MANAGEMENT SYSTEM (AMS) - HIGH PERFORMANCE UNIT
Date: 2026-09-26
Athlete Name: Arjun Patel
Player Identifier: Player 9
Squad: Elite Men's National Squad / Red Ball Pace Unit
Role: Strike Right-Arm Fast Bowler (Release Speed: 142-147 km/h)
Lead Physiotherapist: Dr. Kieran Vance, M.Phty, B.Sc (Sports Med)
Chief Medical Officer: Dr. Evelyn Reed, FACSP, MBBS

--------------------------------------------------------------------------------
1. CHIEF COMPLAINT & HISTORY OF PRESENT ILLNESS
Athlete Player 9 reported acute, progressively worsening right-sided lower lumbar pain following Day 3 of the 4-day match fixture on September 24, 2026.
Player bowled a total of 36 overs across 4 days (9 overs on Sept 21, 10 overs on Sept 22, 8 overs on Sept 23, 9 overs on Sept 24).
Player described an aching sensation during early spells that evolved into sharp, focal pain during delivery stride front-foot plant and follow-through.
Aggravated by backward bending, active bowling delivery stride, and running impact.
Subjective morning stiffness lasting >45 minutes. Player took oral ibuprofen (400mg) without symptom resolution.

--------------------------------------------------------------------------------
2. PHYSICAL & CLINICAL EXAMINATION
- Posture & Gait: Antalgic gait favoring right lower limb. Mild flattening of lumbar lordosis.
- Palpation: Extreme focal point tenderness on palpation over the right L4-L5 and L5-S1 pars interarticularis and facet joint capsule. Moderate protective paraspinal muscle spasm on right lumbar erector spinae.
- Range of Motion:
  * Lumbar Flexion: Fingertips to mid-shin (80% normal), mild stretch discomfort.
  * Lumbar Extension: Markedly restricted to 10 degrees (normal ~30 degrees); reproduces acute focal right lower back pain (VAS 8/10).
  * Right Lateral Flexion + Rotation (Quadrant Kemp's test): Severely positive, reproducing sharp local pain.
- Provocation Tests:
  * Stork Test (Single-Leg Hyperextension Test): Standing on right leg with lumbar extension reproduces acute pain at right L5 pars (Positive). Negative when standing on left leg.
  * Active Straight Leg Raise: Negative bilaterally; no dural tension or sciatic nerve root compression.
  * Slump Test: Negative bilaterally. Neurological exam (L2-S1 reflexes, sensation, motor strength) intact.

--------------------------------------------------------------------------------
3. HIGH-SPEED BIOMECHANICAL VIDEO ANALYSIS
Review of high-speed 240fps broadcast telemetry cameras from delivery stride on Sept 23, 2026:
- Delivery Action Classification: Mixed bowling action.
- Pelvic Alignment at Back Foot Contact: Open / semi-front-on (134 degrees).
- Shoulder Alignment at Back Foot Contact: 102 degrees.
- Minimum Shoulder Angle prior to Front Foot Contact: 64 degrees.
- Calculated Shoulder Counter-Rotation: 38 degrees (High Pathological Threshold; literature safe limit <25 degrees).
- Front Foot Contact: Rigid front knee plant with 7.8x body weight peak ground reaction force and excessive trunk lateral flexion (41 degrees contralateral lean).
- Biomechanical Summary: Excessive shoulder counter-rotation (38 deg) combined with lateral flexion creates severe shear torsion across the contralateral right L5 pars interarticularis.

--------------------------------------------------------------------------------
4. IMAGING DIAGNOSTICS (3T MRI LUMBAR SPINE)
Scanned: Sept 25, 2026 at Apex Orthopaedic Imaging Centre
Sequences: T1-weighted, T2-weighted, and STIR fat-suppressed sagittal and axial oblique planes.
Findings:
- Right L5 Pars Interarticularis: Conspicuous, high-intensity STIR fluid signal (hyperintensity) within the bone marrow of the right L5 neural arch.
- Cortical Integrity: Cortical margins of the pars remain intact without full cortical fracture separation (Grade 1 active stress reaction / early pre-spondylolytic edema according to Hollenberg classification).
- Intervertebral Discs: Mild L5-S1 disc dehydration, no herniation or canal stenosis.
- Impression: Acute Grade 1 bone stress reaction of the right L5 pars interarticularis secondary to mechanical overload and mixed bowling action kinematics.

--------------------------------------------------------------------------------
5. GPS TELEMETRY & WORKLOAD CORRELATION
- Chronic Workload Baseline (Past 28 days): 16.2 overs/week.
- Acute Workload Spike (Past 7 days): 36.0 overs bowled.
- Acute:Chronic Workload Ratio (ACWR): 1.72 (Danger Zone > 1.50).
- Catapult PlayerLoad: Day 21-24 combined PlayerLoad exceeded 2,450 units.
- Summary: Clear uncoupling of acute workload from chronic physical readiness. The spike to ACWR 1.72 directly triggered structural bone stress failure.

--------------------------------------------------------------------------------
6. MEDICAL IMPRESSION & CLINICAL MANAGEMENT PLAN
Diagnosis: Acute Right L5 Pars Interarticularis Bone Stress Reaction (Grade 1 Pre-Fracture Edema) with Mixed Action Biomechanical Trigger.
Treatment & Return to Play (RTP) Protocol:
1. Immediate Bowling Deload: Complete cessation of bowling and high-speed running for a minimum of 4 to 6 weeks.
2. Low-Impact Cross-Training: Wattbike cycling in neutral spine posture, seated upper body resistance, and water aerobics.
3. Rehabilitation: Core lumbopelvic stability (transversus abdominis, multifidus activation), gluteal strength, thoracic mobility.
4. Biomechanical Modification: Technical retraining with bowling coach to eliminate shoulder counter-rotation (target <22 degrees) by aligning pelvic approach vectors.
5. Follow-Up: Repeat 3T MRI at Week 6. Graded bowling progression strictly adhering to Gabbett ACWR sweet spot (0.8 - 1.25) before competitive match clearance.
Signed:
Dr. Kieran Vance, Lead Physio
Dr. Evelyn Reed, Chief Sports Physician
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated Physio Notes: {report_path}")


if __name__ == "__main__":
    generate_all_papers()
    generate_player_telemetry()
    generate_physio_notes()
    print("All synthetic datasets and scientific papers generated successfully!")
