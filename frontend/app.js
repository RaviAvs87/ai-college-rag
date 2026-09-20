const t = document.querySelector('.tg');
try { const s = localStorage.getItem('theme'); if (s) document.documentElement.dataset.theme = s } catch (e) { }
t && t.addEventListener('click', () => { const d = document.documentElement, n = d.dataset.theme === 'light' ? 'dark' : 'light'; d.dataset.theme = n; try { localStorage.setItem('theme', n) } catch (e) { } });
