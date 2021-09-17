
<!-- [ SCRIPT ] -->
<script>
	// [ IMPORTS: components ]
	import Header from "./../components/Header.svelte";
	import Analytics from "./../components/Analytics.svelte";
	
	// [ IMPORTS: props ]
	export let is_analysing = false;
	export let analyticsReceived = false;
	export let assessment = 'positive';
	export let ratio = {
        positive: 0,
        neutral: 0,
        negative: 0,
    };
	export let actions = [];


</script>


<!-- [ TEMPLATE: Dashboard > Root ] -->
<div class = 'dashboard'>

	<!-- [ header ] -->
	<Header title="Анализ сообщения" />

	<!-- [ analytics ] -->
	{#if analyticsReceived}
		<Analytics 
			{ assessment }
			{ ratio }
			{ actions }
		/>
	{/if}
	{#if is_analysing}
		<div class="system-message">
			<p class="msg-text">wait ...</p>
		</div>
	{/if}


</div>