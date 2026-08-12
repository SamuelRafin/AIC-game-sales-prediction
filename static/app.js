/* ==========================================================================
   Seller Copilot - Frontend Interactive Logic
   ========================================================================== */

let currentMode = 'game'; // 'game' or 'user'

document.addEventListener('DOMContentLoaded', () => {
  // Initial default load
  executeSearch("Valorant");
});

function switchMode(mode) {
  if (currentMode === mode) return;
  currentMode = mode;

  const btnGame = document.getElementById('btn-mode-game');
  const btnUser = document.getElementById('btn-mode-user');
  const searchInput = document.getElementById('search-input');
  const dropdown = document.getElementById('autocomplete-dropdown');

  dropdown.classList.add('hidden');

  if (mode === 'game') {
    btnGame.classList.add('active');
    btnUser.classList.remove('active');
    searchInput.placeholder = 'Ketik nama game, mis. Valorant';
    searchInput.value = '';
    executeSearch("Valorant");
  } else {
    btnUser.classList.add('active');
    btnGame.classList.remove('active');
    searchInput.placeholder = 'Ketik User ID, mis. 1001';
    searchInput.value = '';
    executeSearch("1001");
  }
}

function handleKeyDown(event) {
  if (event.key === 'Enter') {
    event.preventDefault();
    document.getElementById('autocomplete-dropdown').classList.add('hidden');
    executeSearch();
  }
}

let debounceTimer = null;
function handleInput() {
  clearTimeout(debounceTimer);
  const query = document.getElementById('search-input').value.trim();
  const dropdown = document.getElementById('autocomplete-dropdown');

  if (!query) {
    dropdown.classList.add('hidden');
    return;
  }

  debounceTimer = setTimeout(() => {
    fetch(`/api/autocomplete?type=${currentMode}&q=${encodeURIComponent(query)}`)
      .then(res => res.json())
      .then(data => {
        if (data.results && data.results.length > 0) {
          dropdown.innerHTML = data.results.map(item => `
            <div class="autocomplete-item" onclick="selectAutocomplete('${item.replace(/'/g, "\\'")}')">
              ${item}
            </div>
          `).join('');
          dropdown.classList.remove('hidden');
        } else {
          dropdown.classList.add('hidden');
        }
      })
      .catch(() => dropdown.classList.add('hidden'));
  }, 200);
}

function selectAutocomplete(val) {
  const searchInput = document.getElementById('search-input');
  searchInput.value = val;
  document.getElementById('autocomplete-dropdown').classList.add('hidden');
  executeSearch(val);
}

// Hide dropdown when clicking outside
document.addEventListener('click', (e) => {
  const wrapper = document.querySelector('.search-box-wrapper');
  if (wrapper && !wrapper.contains(e.target)) {
    document.getElementById('autocomplete-dropdown').classList.add('hidden');
  }
});

function executeSearch(overrideQuery) {
  const searchInput = document.getElementById('search-input');
  const query = overrideQuery || searchInput.value.trim() || (currentMode === 'game' ? 'Valorant' : '1001');

  fetch(`/api/search?type=${currentMode}&query=${encodeURIComponent(query)}`)
    .then(res => res.json())
    .then(data => {
      renderDashboard(data);
    })
    .catch(err => {
      console.error("Error fetching search insights:", err);
    });
}

function renderDashboard(data) {
  if (!data || !data.metrics) return;

  // 1. Update Metrics
  document.getElementById('val-skor-kelakuan').textContent = data.metrics.skor_kelakuan || '42%';

  const forecastElem = document.getElementById('val-forecast-4-minggu');
  forecastElem.textContent = (data.metrics.forecast_is_positive ? '↑ ' : '↓ ') + data.metrics.forecast_4_minggu.replace(/^[+↑↓\s]+/, '');
  if (data.metrics.forecast_is_positive) {
    forecastElem.className = 'metric-value metric-green';
  } else {
    forecastElem.className = 'metric-value metric-red';
  }

  const rankLabelElem = document.getElementById('label-metric-rank');
  if (currentMode === 'game') {
    rankLabelElem.textContent = `Rank di genre ${data.metrics.genre_name || 'FPS'}`;
  } else {
    rankLabelElem.textContent = `Rank di segmen ${data.metrics.genre_name || 'User'}`;
  }
  document.getElementById('val-rank-genre').textContent = data.metrics.rank_genre || '#3';

  // 2. Update Card Title
  document.getElementById('results-card-title').textContent = data.section_title || (currentMode === 'game' ? 'Saran bundling' : 'Rekomendasi game selanjutnya');

  // 3. Update Recommendations List
  const listElem = document.getElementById('recommendations-list');
  if (data.recommendations && data.recommendations.length > 0) {
    listElem.innerHTML = data.recommendations.map(item => `
      <div class="recommendation-item">
        <div class="item-header">
          <span class="item-name">${escapeHtml(item.name)}</span>
          <span class="item-percentage">${item.percentage}%</span>
        </div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" style="width: ${item.percentage}%;"></div>
        </div>
      </div>
    `).join('');
  } else {
    listElem.innerHTML = `<div class="item-name" style="color: var(--text-muted);">Tidak ada rekomendasi ditemukan.</div>`;
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
