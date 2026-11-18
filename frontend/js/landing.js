/**
 * RIDEGUARD AI — 3D ANIMATED WHITE & YELLOW SAAS LANDING PAGE
 * Powered by Three.js WebGL 3D Mobility City • GSAP Animations • Interactive Preview
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Lucide Icons
    if (window.lucide) {
        window.lucide.createIcons();
    }

    // 2. Navbar Scroll Blur & State
    initNavbarScroll();

    // 3. Mobile Navigation Drawer Toggle
    initMobileNav();

    // 4. Three.js 3D Interactive Mobility City Scene
    initThreeMobilityScene();

    // 5. Product Preview Interactive Tab Switcher
    initProductPreviewTabs();

    // 6. About Modal Controller
    initAboutModal();

    // 7. GSAP Scroll Animations & 3D Tilt Card Effects
    initGsapAnimations();
});

/* -------------------------------------------------------------
   1. NAVBAR SCROLL EFFECT
   ------------------------------------------------------------- */
function initNavbarScroll() {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;

    window.addEventListener('scroll', () => {
        if (window.scrollY > 40) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    }, { passive: true });
}

/* -------------------------------------------------------------
   2. MOBILE NAVIGATION DRAWER
   ------------------------------------------------------------- */
function initMobileNav() {
    const menuBtn = document.getElementById('mobile-menu-btn');
    const drawer = document.getElementById('mobile-nav-drawer');
    const links = document.querySelectorAll('.mobile-nav-link');

    if (!menuBtn || !drawer) return;

    menuBtn.addEventListener('click', () => {
        drawer.classList.toggle('open');
        if (drawer.classList.contains('open')) {
            menuBtn.innerHTML = '<i data-lucide="x" class="w-6 h-6"></i>';
        } else {
            menuBtn.innerHTML = '<i data-lucide="menu" class="w-6 h-6"></i>';
        }
        if (window.lucide) window.lucide.createIcons();
    });

    links.forEach(link => {
        link.addEventListener('click', () => {
            drawer.classList.remove('open');
            menuBtn.innerHTML = '<i data-lucide="menu" class="w-6 h-6"></i>';
            if (window.lucide) window.lucide.createIcons();
        });
    });
}

/* -------------------------------------------------------------
   3. THREE.JS 3D INTERACTIVE MOBILITY CITY SCENE
   Builds an isometric 3D urban grid with procedural white blocks,
   glowing yellow wireframes, 3D curved transport arteries, moving
   golden vehicle pods, and interactive 3D mouse parallax.
   ------------------------------------------------------------- */
function initThreeMobilityScene() {
    const container = document.querySelector('.canvas-container');
    const canvas = document.getElementById('three-mobility-canvas');
    if (!container || !canvas || typeof THREE === 'undefined') return;

    // Dimensions
    let width = container.clientWidth;
    let height = container.clientHeight;

    // 1. Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xF8FAFC);
    scene.fog = new THREE.FogExp2(0xF8FAFC, 0.015);

    // 2. Camera setup (Isometric perspective)
    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 1000);
    const initialCamPos = { x: 30, y: 38, z: 42 };
    camera.position.set(initialCamPos.x, initialCamPos.y, initialCamPos.z);
    camera.lookAt(0, 0, 0);

    // 3. Renderer setup
    const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // 4. Lighting (Warm White & Yellow Glows)
    const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.75);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xFEF08A, 0.85);
    dirLight.position.set(20, 40, 20);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    scene.add(dirLight);

    // Yellow Dispatch Hub Point Light
    const yellowHubLight = new THREE.PointLight(0xEAB308, 1.8, 45);
    yellowHubLight.position.set(0, 8, 0);
    scene.add(yellowHubLight);

    // Soft Blue Secondary Fill
    const blueFillLight = new THREE.PointLight(0x38BDF8, 0.8, 40);
    blueFillLight.position.set(-15, 6, -15);
    scene.add(blueFillLight);

    // 5. Ground Plane & Grid
    const groundGeo = new THREE.PlaneGeometry(120, 120);
    const groundMat = new THREE.MeshStandardMaterial({
        color: 0xF1F5F9,
        roughness: 0.9,
        metalness: 0.1
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.05;
    ground.receiveShadow = true;
    scene.add(ground);

    // Yellow-accented Grid Helper
    const gridHelper = new THREE.GridHelper(90, 30, 0xEAB308, 0xE2E8F0);
    gridHelper.position.y = 0;
    scene.add(gridHelper);

    // 6. Procedural 3D City Buildings with Yellow Edges
    const buildingGroup = new THREE.Group();
    const buildingMat = new THREE.MeshStandardMaterial({
        color: 0xFFFFFF,
        roughness: 0.25,
        metalness: 0.05
    });
    const edgeMat = new THREE.LineBasicMaterial({ color: 0xFACC15, linewidth: 1.5 });

    const buildingCoords = [
        // Central Terminal Complex
        { x: 0, z: 0, w: 6, d: 6, h: 10, isCentral: true },
        { x: -9, z: -4, w: 4.5, d: 4.5, h: 7 },
        { x: 9, z: -5, w: 5, d: 4, h: 8 },
        { x: -8, z: 8, w: 4, d: 5, h: 6 },
        { x: 8, z: 7, w: 4.5, d: 4.5, h: 7.5 },
        // Outer Peripheral Hubs
        { x: -18, z: -12, w: 5, d: 5, h: 5 },
        { x: 18, z: -14, w: 6, d: 5, h: 6 },
        { x: -16, z: 16, w: 5, d: 6, h: 5.5 },
        { x: 17, z: 15, w: 5.5, d: 5, h: 6.5 },
        { x: 0, z: -18, w: 6, d: 4, h: 4.5 },
        { x: 0, z: 18, w: 5, d: 5, h: 5 }
    ];

    buildingCoords.forEach(b => {
        const geo = new THREE.BoxGeometry(b.w, b.h, b.d);
        const mesh = new THREE.Mesh(geo, buildingMat);
        mesh.position.set(b.x, b.h / 2, b.z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        buildingGroup.add(mesh);

        // Yellow Wireframe Accent Edges
        const edges = new THREE.EdgesGeometry(geo);
        const line = new THREE.LineSegments(edges, edgeMat);
        line.position.copy(mesh.position);
        buildingGroup.add(line);

        // Roof Heli-pad / Antenna on Central Hub
        if (b.isCentral) {
            const padGeo = new THREE.CylinderGeometry(2, 2, 0.4, 24);
            const padMat = new THREE.MeshStandardMaterial({ color: 0xEAB308, roughness: 0.4 });
            const pad = new THREE.Mesh(padGeo, padMat);
            pad.position.set(b.x, b.h + 0.2, b.z);
            buildingGroup.add(pad);
        }
    });
    scene.add(buildingGroup);

    // 7. 3D Curved Arteries & Highway Ribbons
    const routeCurves = [
        new THREE.CatmullRomCurve3([
            new THREE.Vector3(-22, 1.2, -18),
            new THREE.Vector3(-10, 2.5, -6),
            new THREE.Vector3(0, 3.5, 0),
            new THREE.Vector3(12, 2.2, 8),
            new THREE.Vector3(24, 1.2, 18)
        ]),
        new THREE.CatmullRomCurve3([
            new THREE.Vector3(-20, 1.2, 18),
            new THREE.Vector3(-8, 2.2, 8),
            new THREE.Vector3(0, 3.5, 0),
            new THREE.Vector3(10, 2.5, -8),
            new THREE.Vector3(22, 1.2, -18)
        ]),
        // Circular Peripheral Transit Loop
        new THREE.CatmullRomCurve3([
            new THREE.Vector3(0, 1.5, 20),
            new THREE.Vector3(18, 1.5, 12),
            new THREE.Vector3(20, 1.5, -8),
            new THREE.Vector3(0, 1.5, -20),
            new THREE.Vector3(-18, 1.5, -10),
            new THREE.Vector3(-18, 1.5, 12),
            new THREE.Vector3(0, 1.5, 20)
        ])
    ];

    const tubeMat = new THREE.MeshStandardMaterial({
        color: 0xFEF08A,
        roughness: 0.3,
        metalness: 0.1,
        transparent: true,
        opacity: 0.85
    });

    routeCurves.forEach(curve => {
        const tubeGeo = new THREE.TubeGeometry(curve, 64, 0.25, 8, false);
        const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
        scene.add(tubeMesh);
    });

    // 8. 3D Animated Vehicle Pods (Golden Cyber Capsules)
    const vehicles = [];
    const NUM_VEHICLES = 8;
    const vehicleGeo = new THREE.SphereGeometry(0.75, 16, 12);
    const vehicleMat = new THREE.MeshStandardMaterial({
        color: 0xEAB308,
        emissive: 0xCA8A04,
        roughness: 0.2,
        metalness: 0.5
    });

    for (let i = 0; i < NUM_VEHICLES; i++) {
        const mesh = new THREE.Mesh(vehicleGeo, vehicleMat);
        mesh.scale.set(1.4, 0.7, 0.8);
        mesh.castShadow = true;
        scene.add(mesh);

        // Small headlight
        const light = new THREE.PointLight(0xFFFFFF, 0.8, 6);
        mesh.add(light);

        vehicles.push({
            mesh: mesh,
            curveIndex: i % routeCurves.length,
            progress: (i / NUM_VEHICLES),
            speed: 0.0015 + (Math.random() * 0.001)
        });
    }

    // 9. Expanding 3D Radar Pulse Wave
    const pulseRingGeo = new THREE.RingGeometry(1, 1.4, 32);
    const pulseRingMat = new THREE.MeshBasicMaterial({
        color: 0xEAB308,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.7
    });
    const pulseRing = new THREE.Mesh(pulseRingGeo, pulseRingMat);
    pulseRing.rotation.x = -Math.PI / 2;
    pulseRing.position.y = 0.1;
    scene.add(pulseRing);

    let pulseRadius = 1;

    // 10. Mouse Interaction & Parallax Tilt
    let mouseX = 0;
    let mouseY = 0;
    let targetCamX = initialCamPos.x;
    let targetCamY = initialCamPos.y;

    container.addEventListener('mousemove', (e) => {
        const rect = container.getBoundingClientRect();
        mouseX = ((e.clientX - rect.left) / width) * 2 - 1;
        mouseY = -(((e.clientY - rect.top) / height) * 2 - 1);

        targetCamX = initialCamPos.x + (mouseX * 12);
        targetCamY = initialCamPos.y + (mouseY * 8);
    });

    container.addEventListener('mouseleave', () => {
        targetCamX = initialCamPos.x;
        targetCamY = initialCamPos.y;
    });

    // Click effect: Spawn dynamic pulse ring
    container.addEventListener('click', () => {
        pulseRadius = 1;
        pulseRing.material.opacity = 0.9;
        yellowHubLight.intensity = 3.5;
        setTimeout(() => { yellowHubLight.intensity = 1.8; }, 300);
    });

    // Resize Handler
    window.addEventListener('resize', () => {
        if (!container) return;
        width = container.clientWidth;
        height = container.clientHeight;
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    });

    // 11. Main 3D Animation Loop
    function animate3D() {
        requestAnimationFrame(animate3D);

        // Smooth camera movement (Lerp)
        camera.position.x += (targetCamX - camera.position.x) * 0.04;
        camera.position.y += (targetCamY - camera.position.y) * 0.04;
        camera.lookAt(0, 2, 0);

        // Gently rotate city group
        buildingGroup.rotation.y += 0.0006;

        // Animate vehicles along curves
        vehicles.forEach(v => {
            v.progress += v.speed;
            if (v.progress > 1) v.progress = 0;

            const curve = routeCurves[v.curveIndex];
            const point = curve.getPointAt(v.progress);
            v.mesh.position.copy(point);

            // Orient vehicle along tangent
            const tangent = curve.getTangentAt(v.progress);
            v.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), tangent);
        });

        // Animate Radar Pulse
        pulseRadius += 0.16;
        if (pulseRadius > 35) {
            pulseRadius = 1;
        }
        pulseRing.scale.set(pulseRadius, pulseRadius, 1);
        pulseRing.material.opacity = Math.max(0, 0.8 - (pulseRadius / 35));

        renderer.render(scene, camera);
    }

    animate3D();
}

/* -------------------------------------------------------------
   4. PRODUCT PREVIEW INTERACTIVE TABS
   ------------------------------------------------------------- */
function initProductPreviewTabs() {
    const tabButtons = document.querySelectorAll('.window-tab-btn');
    const viewPanels = document.querySelectorAll('.preview-view-panel');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-view');

            // Deactivate all
            tabButtons.forEach(b => b.classList.remove('active'));
            viewPanels.forEach(p => p.classList.remove('active'));

            // Activate clicked
            btn.classList.add('active');
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
        });
    });
}

/* -------------------------------------------------------------
   5. ABOUT MODAL CONTROLLER
   ------------------------------------------------------------- */
function initAboutModal() {
    const modal = document.getElementById('about-modal');
    const openBtns = document.querySelectorAll('.btn-open-about');
    const closeBtn = document.getElementById('btn-close-about');

    if (!modal) return;

    openBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';
        });
    });

    const closeModal = () => {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    };

    if (closeBtn) closeBtn.addEventListener('click', closeModal);

    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal.classList.contains('active')) {
            closeModal();
        }
    });
}

/* -------------------------------------------------------------
   6. GSAP SCROLL & REVEAL ANIMATIONS
   ------------------------------------------------------------- */
function initGsapAnimations() {
    if (typeof gsap === 'undefined') return;

    // Register ScrollTrigger if available
    if (typeof ScrollTrigger !== 'undefined') {
        gsap.registerPlugin(ScrollTrigger);
    }

    // Hero Entry Animation Timeline
    const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });

    tl.from('.navbar', {
        y: -40,
        opacity: 0,
        duration: 0.8
    })
    .from('.hero-badge-pill', {
        scale: 0.85,
        opacity: 0,
        duration: 0.6
    }, '-=0.4')
    .from('.hero-headline', {
        y: 30,
        opacity: 0,
        duration: 0.9
    }, '-=0.4')
    .from('.hero-subtitle', {
        y: 20,
        opacity: 0,
        duration: 0.8
    }, '-=0.6')
    .from('.hero-actions', {
        y: 20,
        opacity: 0,
        duration: 0.7
    }, '-=0.5')
    .from('.hero-metrics-strip', {
        opacity: 0,
        y: 15,
        duration: 0.6
    }, '-=0.4')
    .from('.hero-visual-card', {
        scale: 0.95,
        opacity: 0,
        duration: 1
    }, '-=0.8');

    // Float animation on Telemetry Chips
    gsap.to('.chip-top-right', {
        y: -8,
        duration: 2.8,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut'
    });

    gsap.to('.chip-bottom-left', {
        y: 8,
        duration: 3.2,
        repeat: -1,
        yoyo: true,
        ease: 'sine.inOut'
    });

    // ScrollTrigger Card Staggers
    if (typeof ScrollTrigger !== 'undefined') {
        // Problem Cards Stagger
        gsap.from('.problem-card', {
            scrollTrigger: {
                trigger: '.problem-grid',
                start: 'top 80%',
                toggleActions: 'play none none none'
            },
            y: 40,
            opacity: 0,
            duration: 0.7,
            stagger: 0.12,
            ease: 'power2.out'
        });

        // Capabilities Stagger
        gsap.from('.capability-card', {
            scrollTrigger: {
                trigger: '.capabilities-grid',
                start: 'top 80%',
                toggleActions: 'play none none none'
            },
            y: 40,
            opacity: 0,
            duration: 0.8,
            stagger: 0.15,
            ease: 'power2.out'
        });

        // Pipeline Steps Stagger
        gsap.from('.pipeline-step-card', {
            scrollTrigger: {
                trigger: '.pipeline-grid',
                start: 'top 80%',
                toggleActions: 'play none none none'
            },
            y: 30,
            opacity: 0,
            duration: 0.6,
            stagger: 0.1,
            ease: 'power2.out'
        });

        // Machine Learning Tech Cards
        gsap.from('.tech-model-card', {
            scrollTrigger: {
                trigger: '.models-grid',
                start: 'top 80%',
                toggleActions: 'play none none none'
            },
            y: 35,
            opacity: 0,
            duration: 0.7,
            stagger: 0.12,
            ease: 'power2.out'
        });
    }
}
