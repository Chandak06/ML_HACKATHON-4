const fs=require('fs'); const path=require('path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE);
(async()=>{
const out=path.resolve('results/completion_audit'); fs.mkdirSync(out,{recursive:true});
const context=await chromium.launchPersistentContext(path.resolve('.cache/completion_browser'),{channel:'msedge',headless:true,viewport:{width:1440,height:1450}});
const page=await context.newPage(); const errors=[];
page.on('pageerror',e=>errors.push(String(e)));
await page.goto('http://127.0.0.1:8511',{waitUntil:'domcontentloaded'});
await page.getByRole('button',{name:'Predict Route',exact:true}).waitFor({timeout:60000});
await page.getByRole('button',{name:'Predict Route',exact:true}).click();
await page.getByText('Observation Top-1 accuracy',{exact:true}).waitFor({timeout:180000});
const frame=page.frameLocator('iframe[title="Predicted route and GPS observations"]');
await frame.locator('.leaflet-container').waitFor({timeout:60000});
await frame.locator('.leaflet-control-layers').waitFor();
await page.frames().find(f=>f!==page.mainFrame()).waitForFunction(()=>Array.from(document.querySelectorAll('.leaflet-tile')).every(img=>img.complete && img.naturalWidth>0));
await page.screenshot({path:path.join(out,'ui_prediction_verified.png'),fullPage:true});
const mapFrame=page.frames().find(f=>f!==page.mainFrame());
const uiMap=await mapFrame.evaluate(()=>({leaflet:typeof L!=='undefined',canvases:document.querySelectorAll('canvas').length,tiles:document.querySelectorAll('.leaflet-tile-loaded').length,layerText:document.querySelector('.leaflet-control-layers').textContent}));
if(!uiMap.leaflet || !uiMap.canvases || /Reference route/.test(uiMap.layerText)) throw Error('UI map integrity check failed');
if(!/Road network/.test(uiMap.layerText)) throw Error('Missing UI network layer');
await page.goto('file:///'+path.join(out,'representative_offline.html').replaceAll('\\','/'),{waitUntil:'load'});
await page.locator('.leaflet-container').waitFor({timeout:60000});
await page.locator('canvas').waitFor();
await page.screenshot({path:path.join(out,'folium_offline.png'),fullPage:true});
const offline=await page.evaluate(()=>({tiles:document.querySelectorAll('.leaflet-tile-loaded').length,layerText:document.querySelector('.leaflet-control-layers').textContent}));
await page.goto('http://127.0.0.1:8011/docs',{waitUntil:'domcontentloaded'});
await page.getByRole('heading',{name:'PathSnap',exact:false}).first().waitFor({timeout:30000});
await page.screenshot({path:path.join(out,'api_docs.png'),fullPage:true});
fs.writeFileSync(path.join(out,'browser_verification.json'),JSON.stringify({ui_url:'http://127.0.0.1:8511',trip_id:'00000000',ui_map:uiMap,offline_map:offline,page_errors:errors,cross_laptop_verified:false},null,2));
console.log(JSON.stringify({uiMap,offline,errors}));
await context.close();
})().catch(e=>{console.error(e);process.exitCode=1});


