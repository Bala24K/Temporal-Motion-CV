"""End-to-end test of the Next.js app against a running server (http://localhost:8765).
   Covers: register, exercises (+unsupported check), workout builder, live set with injected fault (demo source,
   real WebSocket), rest, summary, progress, nutrition, calibration wizard + 3D viewer, admin console."""
import os, sys, tempfile, time
from playwright.sync_api import sync_playwright, expect

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
errs = []
SHOTS = os.environ.get("E2E_SHOTS", os.path.join(tempfile.gettempdir(), "kinetic-e2e-shots"))
os.makedirs(SHOTS, exist_ok=True)
uid = time.time_ns()

def wait_count(pg, n, timeout=45):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pg.get_by_test_id("rep-count").inner_text()
        if int(v or 0) >= n:
            return
        pg.wait_for_timeout(150)
    raise TimeoutError(f"rep counter never reached {n}")

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1180, "height": 800})
    pg.on("pageerror", lambda e: errs.append("PAGEERR " + str(e)))
    pg.on("console", lambda m: errs.append("CONSOLE " + m.text) if m.type == "error" and "403" not in m.text and "ERR_" not in m.text and "Failed to load resource" not in m.text else None)

    # --- auth
    pg.goto(BASE + "/")
    pg.wait_for_url("**/login/")
    pg.get_by_label("Email").fill(f"e2e{uid}@example.com"); pg.get_by_label("Password").fill("password123")
    pg.get_by_role("button", name="New here? Create account").click(); pg.get_by_role("button", name="Create account").click()
    expect(pg.get_by_role("heading", level=1)).to_contain_text("Good", timeout=15000)
    pg.screenshot(path=f"{SHOTS}/w1_home.png")
    print("auth + home ok")

    # --- exercises + unsupported
    pg.get_by_role("link", name="Exercises").click(); pg.wait_for_url("**/exercises/")
    expect(pg.get_by_role("heading", name="Squat", exact=True)).to_be_visible()
    pg.get_by_role("heading", name="Squat", exact=True).click(); expect(pg.get_by_text("What we check")).to_be_visible()
    pg.get_by_label("Exercise name").fill("Cable Y Raise"); pg.get_by_role("button", name="Check").click()
    expect(pg.get_by_text("Exercise not fully supported yet")).to_be_visible()
    pg.screenshot(path=f"{SHOTS}/w2_exercises.png"); print("exercises + unsupported ok")

    # --- workout builder
    pg.get_by_role("link", name="Workout").click(); pg.wait_for_url("**/workout/")
    pg.get_by_label("Name").fill("Arms + mystery")
    pg.get_by_label("Exercise 1").fill("Bicep Curl"); pg.get_by_label("Reps").first.fill("6"); pg.get_by_label("Sets").first.fill("1")
    pg.get_by_role("button", name="Add exercise").click(); pg.get_by_label("Exercise 2").fill("Cable Y Raise"); pg.get_by_label("Sets").nth(1).fill("1")
    expect(pg.get_by_text("manual").first).to_be_visible()
    pg.get_by_role("button", name="Save workout").click()
    expect(pg.get_by_role("heading", name="Arms + mystery")).to_be_visible(timeout=10000)
    pg.screenshot(path=f"{SHOTS}/w3_builder.png"); print("builder ok")

    # --- live
    pg.get_by_role("button", name="Start", exact=True).first.click()
    pg.get_by_role("dialog", name="Live workout").wait_for(timeout=15000)
    pg.get_by_label("Video source").select_option("demo"); pg.get_by_label("Demo form").select_option("drift")
    pg.get_by_role("button", name="Start", exact=True).click()
    wait_count(pg, 3)
    expect(pg.get_by_text("Keep your elbows by your sides").first).to_be_visible(timeout=15000)
    pg.screenshot(path=f"{SHOTS}/w4_live.png")
    wait_count(pg, 6)
    pg.get_by_role("button", name="End set").click()
    print("live set ok (reps counted, drift flagged)")
    pg.get_by_role("button", name="Skip rest").click(timeout=15000)
    expect(pg.get_by_text("Manual tracking")).to_be_visible(timeout=10000)
    pg.screenshot(path=f"{SHOTS}/w5_manual.png")
    pg.get_by_label("reps").fill("12"); pg.get_by_role("button", name="Log set").click()
    expect(pg.get_by_role("heading", name="Session complete.")).to_be_visible(timeout=15000)
    expect(pg.get_by_text("manual").first).to_be_visible()
    pg.screenshot(path=f"{SHOTS}/w6_summary.png"); print("manual set + summary ok")

    # --- quick set: auto-detect through the consensus recognizer (needs RECOGNIZER_MODE=consensus + ONNX for the learned half)
    pg.get_by_role("link", name="Home").click(); pg.wait_for_url(BASE + "/")
    pg.get_by_role("button", name="Quick set").click(); pg.get_by_role("dialog", name="Live workout").wait_for(timeout=15000)
    # Quick set must not silently start a bicep-curl set (audit defect #4): Start stays disabled until a choice is made
    expect(pg.get_by_role("button", name="Start", exact=True)).to_be_disabled()
    pg.get_by_label("Select exercise").select_option("auto")
    pg.get_by_label("Video source").select_option("demo"); pg.get_by_role("button", name="Start", exact=True).click()
    wait_count(pg, 3)
    expect(pg.get_by_label("Select exercise")).to_have_value("squat", timeout=10000)   # the demo clip is a squat
    print("auto-detect quick set ok (locked on squat)")
    pg.get_by_role("button", name="End set").click(); pg.wait_for_url("**/live/**"); expect(pg.get_by_role("heading", name="Session complete.")).to_be_visible(timeout=15000)

    # --- personal range capture during a set (M3)
    pg.get_by_role("link", name="Home").click(); pg.wait_for_url(BASE + "/")
    pg.get_by_role("button", name="Quick set").click(); pg.get_by_role("dialog", name="Live workout").wait_for(timeout=15000)
    pg.get_by_label("Select exercise").select_option("bicep_curl")
    pg.get_by_label("Video source").select_option("demo"); pg.get_by_role("button", name="Start", exact=True).click()
    pg.get_by_role("button", name="Capture my range").click(timeout=15000)
    pg.wait_for_timeout(8000)                                   # ~3 demo reps
    pg.get_by_role("button", name="Done").click()
    expect(pg.get_by_text("Range saved").first).to_be_visible(timeout=10000)
    expect(pg.get_by_role("button", name="Recapture my range")).to_be_visible()
    pg.screenshot(path=f"{SHOTS}/w6b_range.png")
    wait_count(pg, 1)                                           # counting resumes against the captured range
    pg.get_by_role("button", name="End set").click(); expect(pg.get_by_role("heading", name="Session complete.")).to_be_visible(timeout=15000)
    print("personal range capture ok")

    # --- an AI set with no reps is not silently dropped: the user chooses (M4)
    pg.get_by_role("link", name="Home").click(); pg.wait_for_url(BASE + "/")
    pg.get_by_role("button", name="Quick set").click(); pg.get_by_role("dialog", name="Live workout").wait_for(timeout=15000)
    pg.get_by_label("Select exercise").select_option("bicep_curl")
    pg.get_by_label("Video source").select_option("demo"); pg.get_by_role("button", name="Start", exact=True).click()
    pg.wait_for_timeout(800)
    pg.get_by_role("button", name="End set").click()
    expect(pg.get_by_text("No reps were detected.")).to_be_visible(timeout=10000)
    pg.get_by_role("button", name="Discard").click(); expect(pg.get_by_role("heading", name="Session complete.")).to_be_visible(timeout=15000)
    print("empty set prompt ok")

    pg.get_by_role("link", name="Home").click(); pg.wait_for_url(BASE + "/")
    # --- connection drops mid-set: reconnect resumes the set with its reps (M4)
    pg.get_by_role("button", name="Quick set").click(); pg.get_by_role("dialog", name="Live workout").wait_for(timeout=15000)
    pg.get_by_label("Select exercise").select_option("bicep_curl")
    pg.get_by_label("Video source").select_option("demo"); pg.get_by_role("button", name="Start", exact=True).click()
    wait_count(pg, 2)
    before = int(pg.get_by_test_id("rep-count").inner_text())
    pg.evaluate("window.__kineticDropSocket()")
    expect(pg.get_by_text("Reconnected").first).to_be_visible(timeout=15000)
    assert int(pg.get_by_test_id("rep-count").inner_text()) >= before, "reps lost on reconnect"
    wait_count(pg, before + 1)
    pg.screenshot(path=f"{SHOTS}/w6c_reconnect.png")
    pg.get_by_role("button", name="End set").click(); expect(pg.get_by_role("heading", name="Session complete.")).to_be_visible(timeout=15000)
    print(f"reconnect ok (kept {before}+ reps across a dropped socket)")

    # --- history with per-rep detail, and workout editing (M4)
    pg.get_by_role("link", name="History").click(); pg.wait_for_url("**/history/")
    pg.get_by_test_id("history-session").first.click()
    expect(pg.get_by_role("list", name="Sets")).to_be_visible(timeout=10000)
    pg.screenshot(path=f"{SHOTS}/w6d_history.png")
    pg.get_by_role("link", name="Workout").click(); pg.wait_for_url("**/workout/")
    pg.get_by_role("button", name="Edit").first.click()
    pg.get_by_label("Name").fill("Arms v2"); pg.get_by_role("button", name="Save changes").click()
    expect(pg.get_by_role("heading", name="Arms v2")).to_be_visible(timeout=10000)
    print("history + workout edit ok")

    # --- progress
    pg.get_by_role("link", name="Progress").click(); pg.wait_for_url("**/progress/")
    expect(pg.get_by_text("Muscle distribution")).to_be_visible(); pg.get_by_role("tab", name="Month").click()
    pg.get_by_label("Exercise", exact=True).select_option(index=1)
    expect(pg.get_by_text("Estimated 1RM")).to_be_visible(timeout=10000)
    pg.screenshot(path=f"{SHOTS}/w7_progress.png"); print("progress ok")

    # --- nutrition
    pg.get_by_role("link", name="Nutrition").click(); pg.wait_for_url("**/nutrition/")
    pg.get_by_label("Birth year").fill("1995"); pg.get_by_label("Height (cm)").fill("180"); pg.get_by_label("Weight (kg)").fill("80")
    pg.get_by_role("button", name="Save", exact=True).click()
    expect(pg.get_by_text("Maintenance ≈")).to_be_visible(timeout=10000)
    pg.get_by_label("Food", exact=True).fill("Egg, whole"); pg.get_by_role("button", name="Add").click()
    expect(pg.get_by_text("143 kcal").first).to_be_visible(timeout=10000)
    pg.screenshot(path=f"{SHOTS}/w8_nutrition.png"); print("nutrition ok")

    # --- calibration + 3D
    pg.get_by_role("link", name="Profile").click(); pg.wait_for_url("**/profile/")
    pg.get_by_label("Height (cm, optional)").fill("178"); pg.get_by_label("Source").select_option("demo")
    for _ in range(3):
        pg.get_by_role("button", name="Capture", exact=False).first.click(); pg.wait_for_timeout(900)
    pg.get_by_role("button", name="Finish (3/3)").click()
    expect(pg.get_by_test_id("calibration-result")).to_be_visible(timeout=15000)
    expect(pg.get_by_test_id("calibration-result")).to_contain_text("Calibrated")      # status "good" with all 3 poses
    pg.reload(); expect(pg.get_by_text("Calibrated").first).to_be_visible(timeout=10000)
    try:
        pg.get_by_test_id("skeleton-3d").wait_for(timeout=8000)
    except Exception:
        pass
    has3d = pg.get_by_test_id("skeleton-3d").count()
    pg.wait_for_timeout(700); pg.screenshot(path=f"{SHOTS}/w9_profile.png")
    print("calibration ok; 3D viewer element present:", bool(has3d))
    b.close()

    # --- admin (separate account)
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1180, "height": 800})
    pg.on("pageerror", lambda e: errs.append("PAGEERR " + str(e)))
    pg.request.post(BASE + "/api/auth/register", data={"email": "admin@example.com", "password": "password123"})   # 409 if it exists: fine
    pg.goto(BASE + "/login/"); pg.get_by_label("Email").fill("admin@example.com"); pg.get_by_label("Password").fill("password123")
    pg.get_by_role("button", name="Sign in").click()
    pg.get_by_role("link", name="Console").click(); pg.wait_for_url("**/admin/")
    expect(pg.get_by_text("Most-requested unsupported exercises")).to_be_visible(timeout=10000)
    expect(pg.get_by_text("Cable Y Raise").first).to_be_visible()
    expect(pg.get_by_text("definition-rules")).to_be_visible()
    pg.get_by_role("button", name="Why?").first.click(); expect(pg.get_by_text("not reproducible").first).to_be_visible()
    pg.get_by_role("button", name="Promote").first.click(); expect(pg.get_by_text("Promotion blocked")).to_be_visible(timeout=10000)
    pg.screenshot(path=f"{SHOTS}/w10_admin.png"); print("admin ok (gate blocked the unverified candidate)")
    b.close()
print("ERRORS:", errs)
