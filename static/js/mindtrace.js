/* MindTrace NG interactions (vanilla JS) */
(() => {
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const one = (s, el = document) => el.querySelector(s);
  const all = (s, el = document) => Array.from(el.querySelectorAll(s));

  // public menu
  const burger = one('.burger'), links = one('#toplinks');
  burger?.addEventListener('click', () => burger.setAttribute('aria-expanded', links.classList.toggle('open')));
  links?.addEventListener('click', e => { if (e.target.closest('a')) { links.classList.remove('open'); burger?.setAttribute('aria-expanded', 'false'); } });

  // count-up numbers
  const count = el => {
    const raw = el.dataset.n, end = parseFloat(raw), dp = (raw.split('.')[1] || '').length;
    if (still || !isFinite(end)) return;
    const t0 = performance.now();
    const frame = t => {
      const k = Math.min((t - t0) / 1100, 1), e = 1 - (1 - k) ** 3;
      el.textContent = dp ? (end * e).toFixed(dp) : Math.round(end * e).toLocaleString('en-NG');
      if (k < 1) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  };

  // reveal on scroll
  const seen = new IntersectionObserver(list => list.forEach(en => {
    if (!en.isIntersecting) return;
    const el = en.target;
    setTimeout(() => el.classList.add('in'), (el.dataset.delay || 0) * 1);
    all('[data-n]', el).forEach(count);
    if (el.matches('[data-n]')) count(el);
    seen.unobserve(el);
  }), { threshold: 0.15 });
  all('.flow-row.rv').forEach((el, i) => { el.dataset.delay = i * 70; });
  all('.rv, .outcome, .kpi, .board').forEach(el => seen.observe(el));
  all('.col-bar').forEach((el, i) => el.style.setProperty('--o', i));

  // tabs (report sections, zone list)
  all('[role=tablist]').forEach(list => {
    const tabs = all('[role=tab]', list);
    const pick = (tab, focus) => {
      tabs.forEach(t => {
        const on = t === tab;
        t.setAttribute('aria-selected', on); t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
      });
      if (focus) tab.focus();
    };
    tabs.forEach((t, i) => {
      t.addEventListener('click', () => pick(t));
      t.addEventListener('keydown', e => {
        const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
        if (d) { e.preventDefault(); pick(tabs[(i + d + tabs.length) % tabs.length], true); }
      });
    });
  });

  // hero viewer tilt
  const tilt = one('[data-tilt]');
  if (tilt && !still && matchMedia('(hover: hover)').matches) {
    tilt.addEventListener('pointermove', e => {
      const r = tilt.getBoundingClientRect();
      tilt.style.setProperty('--ry', `${((e.clientX - r.left) / r.width - .5) * 7}deg`);
      tilt.style.setProperty('--rx', `${(.5 - (e.clientY - r.top) / r.height) * 7}deg`);
    });
    tilt.addEventListener('pointerleave', () => { tilt.style.setProperty('--rx', '0deg'); tilt.style.setProperty('--ry', '0deg'); });
  }

  // small behaviours
  all('.snack').forEach(s => { one('button', s).addEventListener('click', () => s.remove()); setTimeout(() => s.remove(), 6000); });
  all('form[data-busy]').forEach(f => f.addEventListener('submit', () => { const b = one('[type=submit]', f); b.disabled = true; const s = one('span', b); if (s) s.textContent = 'One moment'; }));
  all('form[data-confirm]').forEach(f => f.addEventListener('submit', e => { if (!confirm(f.dataset.confirm)) e.preventDefault(); }));
  all('[data-print]').forEach(b => b.addEventListener('click', () => print()));
  all('[data-autosubmit]').forEach(s => s.addEventListener('change', () => s.form.submit()));
  all('tr[data-href]').forEach(r => r.addEventListener('click', e => { if (!e.target.closest('a')) location.href = r.dataset.href; }));

  // scan upload
  const form = one('#scan-form');
  if (form) {
    const zone = one('#zone'), input = one('#mri'), empty = one('.zone-empty'), filled = one('.zone-file');
    const warn = one('#client-error'), busy = one('#busy'), steps = all('.stepper li'), ref = one('#scan_ref');
    const types = ['png', 'jpg', 'jpeg', 'bmp'];
    const step = n => steps.forEach(s => s.classList.toggle('on', +s.dataset.step <= n));
    const say = msg => { warn.hidden = !msg; one('span', warn).textContent = msg || ''; one('#server-error')?.remove(); };
    const reset = () => { input.value = ''; filled.hidden = true; empty.hidden = false; zone.classList.remove('filled'); step(1); };
    const take = file => {
      if (!file) return reset();
      if (!types.includes(file.name.split('.').pop().toLowerCase())) { reset(); return say('Unsupported file type. Upload a PNG, JPG or BMP image.'); }
      if (file.size > 8 * 1024 * 1024) { reset(); return say('The file is larger than 8 MB.'); }
      say('');
      one('#thumb').src = URL.createObjectURL(file);
      one('#fname').textContent = file.name;
      one('#fsize').textContent = file.size >= 1048576 ? (file.size / 1048576).toFixed(1) + ' MB' : Math.ceil(file.size / 1024) + ' KB';
      empty.hidden = true; filled.hidden = false; zone.classList.add('filled'); step(2);
    };
    input.addEventListener('change', () => take(input.files[0]));
    one('#swap').addEventListener('click', () => { reset(); input.click(); });
    ref.addEventListener('input', () => { if (input.files.length) step(ref.value.trim() ? 3 : 2); });
    ['dragenter', 'dragover'].forEach(n => zone.addEventListener(n, e => { e.preventDefault(); zone.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach(n => zone.addEventListener(n, e => { e.preventDefault(); zone.classList.remove('drag'); }));
    zone.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (!f) return; const dt = new DataTransfer(); dt.items.add(f); input.files = dt.files; take(f); });
    form.addEventListener('submit', e => {
      if (!input.files.length) { e.preventDefault(); return say('Choose an MRI image first.'); }
      step(3); one('#go').disabled = true; busy.hidden = false;
      const msgs = ['Checking the image', 'Preparing 128 × 128 grayscale', 'Measuring texture and intensity', 'Collecting the forest vote'];
      const bar = one('.bar i', busy), label = one('#busy-step');
      msgs.forEach((m, i) => setTimeout(() => { label.textContent = m; bar.style.width = `${(i + 1) * 24}%`; }, i * 350));
    });
    addEventListener('pageshow', e => { if (e.persisted) { busy.hidden = true; one('#go').disabled = false; } });
  }
})();
