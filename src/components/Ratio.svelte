
<!-- [ SCRIPT ] -->
<script>
	// [ IMPORTS: components ]
	// import Assessment from "./components/Assessment.svelte";
	// import Ratio from "./components/Ratio.svelte";
	// import Actions from "./components/Actions.svelte";
	
	// [ IMPORTS: props ]
	export let ratio = {
        positive: 0,
        neutral: 0,
        negative: 0,
    };

</script>


<!-- [ TEMPLATE: Assessment > Analytics ] -->
<div class = 'section-ratio'>

	<h2 class = 'section-title'>Степень эмоциональности</h2>

    <div class="content">
        <!-- progressbar -->
        <div class="progressbar">
            {#if ratio.positive > 0}
                <div class="bar bar-green" style="width: { ratio.positive }%"></div>
            {/if}
            {#if ratio.neutral > 0}
                <div class="bar bar-yellow" style="width: { ratio.neutral }%"></div>
            {/if}
            {#if ratio.negative > 0}
                <div class="bar bar-red" style="width: { ratio.negative }%"></div>   
            {/if}
        </div>

        <!-- infographics -->
        <div class="infographics">
            {#if ratio.positive > 0}
                <div class="info-bar info-bar-green">{ ratio.positive }%</div>
            {/if}
            {#if ratio.neutral > 0}
                <div class="info-bar info-bar-yellow">{ ratio.neutral }%</div>
            {/if}
            {#if ratio.negative > 0}
                <div class="info-bar info-bar-red">{ ratio.negative }%</div>   
            {/if}
        </div>
    </div>

</div>